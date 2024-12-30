# Copied from https://github.com/huggingface/diffusers/blob/v0.32.1/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L175
# with the following modifications:
# - It uses SMC to generate samples with high return.
# - It returns all the intermediate latents of the denoising process as well as other SMC-related values for debugging purpose.

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import math
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline, StableDiffusionXLPipelineOutput
from diffusers.utils import deprecate
from .ddim_with_logprob import get_variance, ddim_step_with_mean, ddim_step_with_logprob, ddim_prediction_with_logprob
from torch.utils.checkpoint import checkpoint
from das.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights, adaptive_tempering

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def pipeline_using_smc_sdxl(
    self: StableDiffusionXLPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    eta: float = 1.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[Any] = None,  # PipelineImageInput
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    # SMC parameters
    num_particles: int = 4,
    batch_p: int = 1, # number of particles to run parallely
    resample_strategy: str = "ssp",
    ess_threshold: float = 0.5,
    tempering: str = "schedule",
    tempering_schedule: Union[float, int, str] = "exp",
    tempering_gamma: float = 1.,
    tempering_start: float = 0.,
    reward_fn: Callable[Union[torch.Tensor, np.ndarray], float] = None,
    kl_coeff: float = 1.,
    verbose: bool = False, # True for debugging SMC procedure
    **kwargs,
):
    # Handle deprecated callback parameters
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)
    if callback is not None:
        deprecate("callback", "1.0.0", "Use callback_on_step_end instead")
    if callback_steps is not None:
        deprecate("callback_steps", "1.0.0", "Use callback_on_step_end instead")
    
    # Enable gradient checkpointing for UNet and VAE
    self.unet.enable_gradient_checkpointing()
    self.vae.enable_gradient_checkpointing()

    # Get dtype from the pipeline's unet
    dtype = self.unet.dtype

    # 0. Default height and width
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs
    assert num_particles >= batch_p, "num_particles should be greater than or equal to batch_p"

    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_steps,
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    self.to(device)

    # Move any input tensors to the correct device
    if prompt_embeds is not None:
        prompt_embeds = prompt_embeds.to(device)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device)
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    if negative_pooled_prompt_embeds is not None:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
    if latents is not None:
        latents = latents.to(device)
    if ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = [embed.to(device) for embed in ip_adapter_image_embeds]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=batch_size*batch_p,  # We handle multiple samples via particles
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        clip_skip=clip_skip,
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    prop_latents = self.prepare_latents(
        batch_size * num_particles,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if self.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

    add_time_ids = self._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = self._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * batch_p, 1)

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_particles,
            do_classifier_free_guidance,
        )

    def _pred_noise(latents, t):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = image_embeds

        noise_pred = checkpoint(self.unet,latent_model_input, t, prompt_embeds, None, None, None, None, added_cond_kwargs, use_reentrant=False)[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

        guidance = noise_pred - noise_pred_uncond if do_classifier_free_guidance else torch.zeros_like(noise_pred, device=noise_pred.device)
        return noise_pred, guidance

    def _decode(latents):

        # Handle upcasting
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self.vae.dtype:
            if torch.backends.mps.is_available():
                self.vae = self.vae.to(latents.dtype)
            else:
                latents = latents.to(self.vae.dtype)

        # Handle latents normalization
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None

        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor
        
        # Try decoding with gradient checkpointing
        try:
            image = checkpoint(self.vae.decode, latents, use_reentrant=False).sample
            if torch.isnan(image).any():
                print("WARNING: NaN values detected in decoded image!")
                print(f"NaN count: {torch.isnan(image).sum().item()}")
        except Exception as e:
            print(f"Error during VAE decode: {str(e)}")
            raise

        # Cleanup if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type='pt', do_denormalize=do_denormalize)

        return image

    # 8. Initialize SMC variables
    noise_pred = torch.zeros_like(prop_latents, device=device, dtype=dtype)
    guidance = torch.zeros_like(prop_latents, device=device, dtype=dtype)
    approx_guidance = torch.zeros_like(prop_latents, device=device, dtype=dtype)
    reward_guidance = torch.zeros_like(prop_latents, device=device, dtype=dtype)
    pred_original_sample = torch.zeros_like(prop_latents, device=device, dtype=dtype)
    scale_factor = torch.zeros(batch_size, device=device)
    min_scale_next = torch.zeros(batch_size, device=device)
    rewards = torch.zeros(prop_latents.shape[0], device=device)
    log_twist_func = torch.zeros(prop_latents.shape[0], device=device)
    log_twist_func_prev = torch.zeros(prop_latents.shape[0], device=device)
    log_Z = torch.zeros(batch_size, device=device)
    log_w = torch.zeros(prop_latents.shape[0], device=device)
    log_prob_diffusion = torch.zeros(prop_latents.shape[0], device=device)
    log_prob_proposal = torch.zeros(prop_latents.shape[0], device=device)
    
    resample_fn = resampling_function(resample_strategy=resample_strategy, ess_threshold=ess_threshold)
    all_latents = []
    all_log_w = []
    all_resample_indices = []
    ess_trace = []
    scale_factor_trace = []
    rewards_trace = []
    manifold_deviation_trace = torch.tensor([], device=device)
    log_prob_diffusion_trace = torch.tensor([], device=device)

    kl_coeff = torch.tensor(kl_coeff, device=device)
    lookforward_fn = lambda r: r / kl_coeff

    start = int(num_inference_steps * tempering_start)

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size * num_particles)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=prop_latents.dtype)

    def _calc_guidance():
        # Initialize local variables that will be populated
        # global noise_pred, guidance, approx_guidance, pred_original_sample, rewards, log_twist_func
        # Becarful with mixed precisions!!

        if (i >= start):
            with torch.enable_grad():
                for idx in range(math.ceil(num_particles / batch_p)):
                    # Force latents to float16 (HalfTensor) to match UNet weights
                    tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].detach().to(dtype=dtype, device=device).requires_grad_(True)

                    # Run noise prediction in float16
                    tmp_noise_pred, tmp_guidance = _pred_noise(tmp_latents, t)

                    # Get original sample prediction in float16
                    tmp_pred_original_sample, _ = ddim_prediction_with_logprob(
                        self.scheduler, tmp_noise_pred, t, tmp_latents, **extra_step_kwargs
                    )

                    tmp_decoded = _decode(tmp_pred_original_sample)
                    tmp_rewards = checkpoint(reward_fn, tmp_decoded, use_reentrant=False) # becareful of inference_dtype of the reward model! (should be same with vae, i.e. float32)

                    # Convert rewards back to float16 for gradient calculation
                    tmp_rewards = tmp_rewards
                    tmp_log_twist_func = lookforward_fn(tmp_rewards)

                    grad_outputs = torch.ones_like(tmp_log_twist_func, device=device)

                    # Calculate gradients
                    tmp_approx_guidance = torch.autograd.grad(
                        outputs=tmp_log_twist_func,
                        inputs=tmp_latents,
                        grad_outputs=grad_outputs,
                        retain_graph=False,
                        create_graph=False
                    )[0].detach()

                    # Store results in float16
                    pred_original_sample[batch_p*idx : batch_p*(idx+1)] = tmp_pred_original_sample.detach().clone()
                    rewards[batch_p*idx : batch_p*(idx+1)] = tmp_rewards.detach().clone()

                    noise_pred[batch_p*idx : batch_p*(idx+1)] = tmp_noise_pred.detach().clone()
                    guidance[batch_p*idx : batch_p*(idx+1)] = tmp_guidance.detach().clone()

                    log_twist_func[batch_p*idx : batch_p*(idx+1)] = tmp_log_twist_func.detach().clone()
                    approx_guidance[batch_p*idx : batch_p*(idx+1)] = tmp_approx_guidance.clone()

            if torch.isnan(log_twist_func).any():
                if verbose:
                    print("NaN in log twist func, changing it to 0")
                log_twist_func[:] = torch.nan_to_num(log_twist_func)
            if torch.isnan(approx_guidance).any():
                if verbose:
                    print("NaN in approx guidance, changing it to 0")
                approx_guidance[:] = torch.nan_to_num(approx_guidance)

        else:
            for idx in range(math.ceil(num_particles / batch_p)):
                with torch.no_grad():
                    tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].to(dtype=dtype, device=device)
                    tmp_noise_pred, tmp_guidance = _pred_noise(tmp_latents, t)
                    noise_pred[batch_p*idx : batch_p*(idx+1)] = tmp_noise_pred.detach().clone()
                    guidance[batch_p*idx : batch_p*(idx+1)] = tmp_guidance.detach().clone()

        if verbose:
            print("Expected rewards of proposals: ", rewards)

    # 10. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            prev_timestep = (
                t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            )
            # to prevent OOB on gather
            prev_timestep = torch.clamp(prev_timestep, 0, self.scheduler.config.num_train_timesteps - 1)

            # Store current latents
            prop_latents = prop_latents.detach()
            latents = prop_latents.clone()
            log_twist_func_prev = log_twist_func.clone()

            _calc_guidance()
            rewards_trace.append(rewards.view(-1, num_particles).max(dim=1)[0].cpu())

            with torch.no_grad():
                if (i >= start):
                    ################### Select Temperature ###################
                    if isinstance(tempering_schedule, (float, int)):
                        min_scale = torch.tensor([min((tempering_gamma * (i - start))**tempering_schedule, 1.)]*batch_size, device=device)
                        min_scale_next = torch.tensor([min(tempering_gamma * (i + 1 - start), 1.)]*batch_size, device=device)
                    elif tempering_schedule == "exp":
                        min_scale = torch.tensor([min((1 + tempering_gamma) ** (i - start) - 1, 1.)]*batch_size, device=device)
                        min_scale_next = torch.tensor([min((1 + tempering_gamma) ** (i + 1 - start) - 1, 1.)]*batch_size, device=device)
                    elif tempering_schedule == "adaptive":
                        min_scale = scale_factor.clone()
                    else:
                        min_scale = torch.ones(batch_size, device=device)
                        min_scale_next = torch.ones(batch_size, device=device)
                    
                    # Apply tempering strategy
                    if tempering == "adaptive" and i > 0 and (min_scale < 1.).any():
                        scale_factor = adaptive_tempering(
                            log_w.view(-1, num_particles),
                            log_prob_diffusion.view(-1, num_particles),
                            log_twist_func.view(-1, num_particles),
                            log_prob_proposal.view(-1, num_particles),
                            log_twist_func_prev.view(-1, num_particles),
                            min_scale=min_scale,
                            ess_threshold=ess_threshold
                        )
                        min_scale_next = scale_factor.clone()
                    elif tempering == "FreeDoM":
                        scale_factor = (guidance ** 2).mean().sqrt() / (approx_guidance ** 2).mean().sqrt()
                        scale_factor = torch.tensor([scale_factor]*batch_size, device=device)
                        min_scale_next = scale_factor.clone()
                    elif tempering == "schedule":
                        scale_factor = min_scale
                    else:
                        scale_factor = torch.ones(batch_size, device=device)

                    scale_factor_trace.append(scale_factor.cpu())

                    if verbose:
                        print("scale factor (lambda_t): ", scale_factor)

                        print("norm of predicted noise: ", (noise_pred**2).mean().sqrt())
                        print("norm of classifier-free guidance: ", (guidance ** 2).mean().sqrt())
                        print("norm of approximate guidance: ", (1-self.scheduler.alphas_cumprod.gather(0, t.cpu()))*(approx_guidance ** 2).mean().sqrt())
                    
                    # Update twist function and guidance
                    log_twist_func *= scale_factor.repeat_interleave(num_particles, dim=0)
                    approx_guidance *= min_scale_next.repeat_interleave(num_particles, dim=0).view([-1] + [1]*(approx_guidance.dim()-1))

                    if verbose:
                        print("norm of approximate guidance multiplied with scale factor: ", (1-self.scheduler.alphas_cumprod.gather(0, t.cpu()))*(approx_guidance ** 2).mean().sqrt())

                    # Calculate weights
                    incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
                    log_w += incremental_log_w
                    log_Z += torch.logsumexp(log_w, dim=-1)

                    # Calculate ESS and store traces
                    ess = [compute_ess_from_log_w(log_w_prompt).item() for log_w_prompt in log_w.view(-1, num_particles)]
                    all_log_w.append(log_w)
                    ess_trace.append(torch.tensor(ess).cpu())

                    # Resampling step
                    resample_indices, is_resampled, log_w = resample_fn(log_w.view(-1, num_particles))
                    log_w = log_w.view(-1)
                    all_resample_indices.append(resample_indices)

                    if verbose:
                        print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
                        print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
                        print("Incremental weight: ", incremental_log_w)
                        print("Estimated log partition function: ", log_Z)
                        print("Effective sample size: ", ess)
                        print("Resampled particles indices: ", resample_indices)

                    # Update variables based on resampling
                    latents = latents.detach().view(-1, num_particles, *latents.shape[1:])[torch.arange(latents.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *latents.shape[1:])
                    noise_pred = noise_pred.view(-1, num_particles, *noise_pred.shape[1:])[torch.arange(noise_pred.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *noise_pred.shape[1:])
                    pred_original_sample = pred_original_sample.view(-1, num_particles, *pred_original_sample.shape[1:])[torch.arange(pred_original_sample.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *pred_original_sample.shape[1:])
                    manifold_deviation_trace = manifold_deviation_trace.view(-1, num_particles, *manifold_deviation_trace.shape[1:])[torch.arange(manifold_deviation_trace.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *manifold_deviation_trace.shape[1:])
                    log_prob_diffusion_trace = log_prob_diffusion_trace.view(-1, num_particles, *log_prob_diffusion_trace.shape[1:])[torch.arange(log_prob_diffusion_trace.size(0)//num_particles).unsqueeze(1), resample_indices].view(-1, *log_prob_diffusion_trace.shape[1:])

                all_latents.append(latents.cpu())

                # Sample from proposal distribution
                
                prev_sample, prev_sample_mean = ddim_step_with_mean(
                    self.scheduler, noise_pred, t, latents, **extra_step_kwargs
                )

                variance = get_variance(self.scheduler, t, prev_timestep)
                variance = eta**2 * _left_broadcast(variance, prev_sample.shape).to(device)
                std_dev_t = variance.sqrt()

                prop_latents = prev_sample + variance * approx_guidance
                manifold_deviation_trace = torch.cat([manifold_deviation_trace, ((variance * approx_guidance * (-noise_pred)).view(num_particles, -1).sum(dim=1).abs() / (noise_pred**2).view(num_particles, -1).sum(dim=1).sqrt()).unsqueeze(1)], dim=1)
                
                log_prob_diffusion = -0.5 * (prop_latents - prev_sample_mean).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                log_prob_diffusion = log_prob_diffusion.sum(dim=tuple(range(1, log_prob_diffusion.ndim)))
                log_prob_proposal = -0.5 * (prop_latents - prev_sample_mean - variance * approx_guidance).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                log_prob_proposal = log_prob_proposal.sum(dim=tuple(range(1, log_prob_proposal.ndim)))
                log_prob_diffusion[:] = torch.nan_to_num(log_prob_diffusion, nan=-1e6)
                log_prob_proposal[:] = torch.nan_to_num(log_prob_proposal, nan=1e6)

                log_prob_diffusion_trace = torch.cat([log_prob_diffusion_trace, (log_prob_diffusion_trace.transpose(0, 1)[-1] + log_prob_diffusion).unsqueeze(1)], dim=1) if i > 0 else log_prob_diffusion.unsqueeze(1)

            # Update progress
            progress_bar.update()

    # Process final image
    latents = prop_latents.detach()
    log_twist_func_prev = log_twist_func.clone()
    
    # Calculate final rewards and weights
    images = []
    for idx in range(math.ceil(num_particles / batch_p)):
        tmp_latents = latents[batch_p*idx : batch_p*(idx+1)]
        tmp_image = _decode(tmp_latents)
        images.append(tmp_image)
        tmp_rewards = reward_fn(tmp_image).to(torch.float32)
        rewards[batch_p*idx : batch_p*(idx+1)] = tmp_rewards
    
    scale_factor = min_scale_next
    log_twist_func = lookforward_fn(rewards)
    
    scale_factor_trace.append(min_scale_next.cpu())
    rewards_trace.append(rewards.view(-1, num_particles).max(dim=1)[0].cpu())
    
    # Final weight update
    log_w += log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
    log_Z += torch.logsumexp(log_w, dim=-1)
    normalized_w = normalize_weights(log_w.view(-1, num_particles), dim=-1).view(-1)
    
    # Combine images and select best
    images = torch.cat(images, dim=0)

    if verbose:
        print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
        print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
        print("Weight: ", log_w)
        print("Estimated log partition function: ", log_Z)
        print("Effective sample size: ", ess)
    
    all_log_w.append(log_w)
    ess_trace.append(torch.tensor(ess).cpu())

    image = images[torch.argmax(log_w)].unsqueeze(0) # return only image with maximum weight
    latent = latents[torch.argmax(log_w)].unsqueeze(0)

    if output_type == 'latent':
        output = latent
    elif output_type == 'pt':
        output = image
    elif output_type == "np":
        image = self.image_processor.pt_to_numpy(image)
        return image
    elif output_type == "pil":
        image = self.image_processor.pt_to_numpy(image)
        return self.image_processor.numpy_to_pil(image)
    else:
        raise NotImplementedError("output type should be eiteher latent, pt, np, or pil")
    
    # Prepare traces for return
    ess_trace = torch.stack(ess_trace, dim=1)
    scale_factor_trace = torch.stack(scale_factor_trace, dim=1)
    rewards_trace = torch.stack(rewards_trace, dim=1)
    manifold_deviation_trace = manifold_deviation_trace[torch.argmax(log_w)].unsqueeze(0).cpu()
    log_prob_diffusion_trace = -log_prob_diffusion_trace[torch.argmax(log_w)].unsqueeze(0).cpu() / (4 * 64 * 64 * math.log(2))

    # Offload models
    self.maybe_free_model_hooks()

    return output, log_w, normalized_w, all_latents, all_log_w, all_resample_indices, ess_trace, scale_factor_trace, rewards_trace, manifold_deviation_trace, log_prob_diffusion_trace