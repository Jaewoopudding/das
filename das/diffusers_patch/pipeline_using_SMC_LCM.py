# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py#L124
# with the following modifications:
# - It uses SMC to generate samples with high return.
# - It returns all the intermediate latents of the denoising process as well as other SMC-related values for debugging purpose.

from typing import Any, Callable, Dict, List, Optional, Union

import math
import torch
import numpy as np
from das.smc_utils import compute_ess_from_log_w, normalize_log_weights, resampling_function, normalize_weights, adaptive_tempering

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .lcm_with_logprob import lcm_step_with_mean, lcm_prediction
def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

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
def pipeline_using_smc_lcm(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    original_inference_steps: int = None,
    timesteps: List[int] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    # num_images_per_prompt: Optional[int] = 1, # use batch_p instead
    eta: float = 1.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    guidance_rescale: float = 0.0,
    # SMC parameters
    num_particles: int = 4,
    batch_p: int = 1, # number of particles to run parallely
    resample_strategy: str = "ssp",
    ess_threshold: float = 0.5,
    tempering: str = "schedule",
    tempering_schedule: Union[float, int, str] = "exp",
    tempering_gamma: float = 1.,
    tempering_start: float = 0.,
    reward_fn: Callable[Union[torch.Tensor, np.ndarray], float] = None, # Ex) lambda images: _fn(images, prompts.repeat_interleave(batch_p, dim=0), metadata.repeat_interleave(batch_p, dim=0))
    kl_coeff: float = 1.,
    verbose: bool = False # True for debugging SMC procedure
):

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    assert num_particles >= batch_p, "num_particles should be greater than or equal to batch_p"
    
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        prompt_embeds,
    )
    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # Get Guidance Scale Embedding
    # NOTE: We use the Imagen CFG formulation that StableDiffusionPipeline uses rather than the original LCM paper
    # CFG formulation, so we need to subtract 1 from the input guidance_scale.
    # LCM CFG formulation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond), (cfg_scale > 0.0 using CFG)
    w = torch.tensor(guidance_scale - 1).repeat(batch_size * batch_p)
    w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
        device=device, dtype=self.unet.dtype
    )

    # 3. Encode input prompt
    lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
    prompt_embeds, _ = self.encode_prompt(
            prompt,
            device,
            batch_size * batch_p,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    prop_latents = self.prepare_latents(
        batch_size*num_particles, # num_images_per_prompt
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    ) # latents sampled from proposal p(x_T)

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop using Sequential Monte Carlo with Twisted Proposal
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

    # Define helper function for predicting noise in SMC sampling
    def _pred_noise(latents, t):

        # predict the noise residual
        noise_pred = self.unet(
            latents,
            t,
            timestep_cond=w_embedding,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=self.cross_attention_kwargs,
            return_dict=False,
        )[0]

        return noise_pred, 0.

    def _decode(latents):
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(
            image, output_type='pt', do_denormalize=do_denormalize
        )

        return image

    # Intialize variables for SMC sampler
    noise_pred = torch.zeros_like(prop_latents, device=device)
    guidance = torch.zeros_like(prop_latents, device=device)
    approx_guidance = torch.zeros_like(prop_latents, device=device)
    reward_guidance = torch.zeros_like(prop_latents, device=device)
    pred_original_sample = torch.zeros_like(prop_latents, device=device)
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

    kl_coeff = torch.tensor(kl_coeff, device=device).to(torch.float32)
    lookforward_fn = lambda r: r / kl_coeff

    start = int(len(timesteps)*tempering_start)

    def _calc_guidance():
        if (i >= start):
            with torch.enable_grad():
                for idx in range(math.ceil(num_particles / batch_p)): 
                    tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].detach().to(torch.float32).requires_grad_(True)
                    # Noise prediction and predicted x_0
                    tmp_noise_pred, tmp_guidance = _pred_noise(tmp_latents, t)

                    tmp_pred_original_sample = lcm_prediction(
                        self.scheduler, tmp_noise_pred, t, tmp_latents, **extra_step_kwargs
                    )

                    # Calculate rewards
                    tmp_rewards = reward_fn(_decode(tmp_pred_original_sample)).to(torch.float32)
                    tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32)
        
                    # Calculate approximate guidance noise for maximizing reward
                    tmp_approx_guidance = torch.autograd.grad(outputs=tmp_log_twist_func, inputs=tmp_latents, grad_outputs=torch.ones_like(tmp_log_twist_func))[0].detach()

                    pred_original_sample[batch_p*idx : batch_p*(idx+1)] = tmp_pred_original_sample.detach().clone()
                    rewards[batch_p*idx : batch_p*(idx+1)] = tmp_rewards.detach().clone()
                    
                    noise_pred[batch_p*idx : batch_p*(idx+1)] = tmp_noise_pred.detach().clone()
                    guidance[batch_p*idx : batch_p*(idx+1)] = tmp_guidance

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
                tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].detach().requires_grad_(True)
                with torch.no_grad():
                    tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].detach()
                    tmp_noise_pred, tmp_guidance = _pred_noise(tmp_latents, t)

                noise_pred[batch_p*idx : batch_p*(idx+1)] = tmp_noise_pred.detach().clone()
                guidance[batch_p*idx : batch_p*(idx+1)] = tmp_guidance

        if verbose:
            print("Expected rewards of proposals: ", rewards)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            prev_timestep = (
                t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            )
            # to prevent OOB on gather
            prev_timestep = torch.clamp(prev_timestep, 0, self.scheduler.config.num_train_timesteps - 1)

            prop_latents = prop_latents.detach()
            latents = prop_latents.clone()
            log_twist_func_prev = log_twist_func.clone() # Used to calculate weight later

            _calc_guidance()
            rewards_trace.append(rewards.view(-1, num_particles).max(dim=1)[0].cpu())

            with torch.no_grad():
                if (i >= start):
                    # Calculate weights for samples from proposal distribution

                    if isinstance(tempering_schedule, float) or isinstance(tempering_schedule, int):
                        min_scale = torch.tensor([min((tempering_gamma * (i - start))**tempering_schedule, 1.)]*batch_size, device=device)
                        min_scale_next = torch.tensor([min(tempering_gamma * (i + 1 - start), 1.)]*batch_size, device=device)
                    elif tempering_schedule == "exp":
                        min_scale = torch.tensor([min((1 + tempering_gamma) ** (i - start) - 1, 1.)]*batch_size, device=device)
                        min_scale_next = torch.tensor([min((1 + tempering_gamma) ** (i + 1 - start) - 1, 1.)]*batch_size, device=device)
                    elif tempering_schedule == "adaptive":
                        min_scale = scale_factor.clone()
                    else:
                        min_scale = torch.tensor([1.]*batch_size, device=device)
                        min_scale_next = torch.tensor([1.]*batch_size, device=device)
                    
                    if tempering == "adaptive" and i > 0 and (min_scale < 1.).any():
                        scale_factor = adaptive_tempering(log_w.view(-1, num_particles), log_prob_diffusion.view(-1, num_particles), log_twist_func.view(-1, num_particles), log_prob_proposal.view(-1, num_particles), log_twist_func_prev.view(-1, num_particles), min_scale=min_scale, ess_threshold=ess_threshold)
                        # if tempering_schedule == "adaptive":
                        min_scale_next = scale_factor.clone()
                    elif tempering == "adaptive" and i == 0:
                        pass
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

                    log_twist_func *= scale_factor.repeat_interleave(num_particles, dim=0)
                    approx_guidance *= min_scale_next.repeat_interleave(num_particles, dim=0).view([-1] + [1]*(approx_guidance.dim()-1))

                    if verbose:
                        print("norm of approximate guidance multiplied with scale factor: ", (1-self.scheduler.alphas_cumprod.gather(0, t.cpu()))*(approx_guidance ** 2).mean().sqrt())

                    incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
                    
                    log_w += incremental_log_w.detach()
                    log_Z += torch.logsumexp(log_w, dim=-1)                    

                    ess = [compute_ess_from_log_w(log_w_prompt).item() for log_w_prompt in log_w.view(-1, num_particles)]

                    all_log_w.append(log_w)
                    ess_trace.append(torch.tensor(ess).cpu())

                    # resample latents and corresponding variables
                    resample_indices, is_resampled, log_w = resample_fn(log_w.view(-1, num_particles))
                    log_w = log_w.view(-1)

                    all_resample_indices.append(resample_indices)
                    # Note: log_w is updated to 0 for batches with is_resampled==True in resample_fn

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
                
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)

                prev_step_index = self.scheduler.step_index + 1
                if prev_step_index < len(self.scheduler.timesteps):
                    prev_timestep = self.scheduler.timesteps[prev_step_index]
                else:
                    prev_timestep = t

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_prev = 1 - alpha_prod_t_prev

                prev_sample, prev_sample_mean = lcm_step_with_mean(self.scheduler, noise_pred, t, latents, **extra_step_kwargs)

                if self.scheduler.step_index != self.scheduler.num_inference_steps - 1:
                    variance = beta_prod_t_prev
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
                else:
                    prop_latents = prev_sample
                    
                # call the callback, if provided
                if i > num_warmup_steps and i % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Weights for Final samples
        latents = prop_latents.detach()
        log_twist_func_prev = log_twist_func.clone()
        image = []
        for idx in range(math.ceil(num_particles / batch_p)):
            tmp_latents = latents[batch_p*idx : batch_p*(idx+1)]
            tmp_image = _decode(tmp_latents).detach()
            image.append(tmp_image)
            tmp_rewards = reward_fn(tmp_image).detach().to(torch.float32)
            rewards[batch_p*idx : batch_p*(idx+1)] = tmp_rewards
        scale_factor = min_scale_next
        log_twist_func[:] = lookforward_fn(rewards)

        scale_factor_trace.append(min_scale_next.cpu())
        rewards_trace.append(rewards.view(-1, num_particles).max(dim=1)[0].cpu())

        log_w += log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev
        log_Z += torch.logsumexp(log_w, dim=-1)
        normalized_w = normalize_weights(log_w.view(-1, num_particles), dim=-1).view(-1)
        ess = [compute_ess_from_log_w(log_w_prompt) for log_w_prompt in log_w.view(-1, num_particles)]
        image = torch.cat(image, dim=0)

        if verbose:
            print("log_prob_diffusion - log_prob_proposal: ", log_prob_diffusion - log_prob_proposal)
            print("log_twist_func - log_twist_func_prev: ", log_twist_func - log_twist_func_prev)
            print("Weight: ", log_w)
            print("Estimated log partition function: ", log_Z)
            print("Effective sample size: ", ess)

        all_log_w.append(log_w)
        ess_trace.append(torch.tensor(ess).cpu())

        progress_bar.update()
        if callback is not None:
            callback(timesteps, 0, latents)

    image = image[torch.argmax(log_w)].unsqueeze(0) # return only image with maximum weight
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
        

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    ess_trace = torch.stack(ess_trace, dim=1)
    scale_factor_trace = torch.stack(scale_factor_trace, dim=1)
    rewards_trace = torch.stack(rewards_trace, dim=1)
    manifold_deviation_trace = manifold_deviation_trace[torch.argmax(log_w)].unsqueeze(0).cpu()
    log_prob_diffusion_trace = - log_prob_diffusion_trace[torch.argmax(log_w)].unsqueeze(0).cpu() / 4 / 64 / 64 / math.log(2) # bits per dimension

    return output, log_w, normalized_w, all_latents, all_log_w, all_resample_indices, ess_trace, scale_factor_trace, rewards_trace, manifold_deviation_trace, log_prob_diffusion_trace
