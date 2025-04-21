# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
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
from .ddim_with_logprob import get_variance, ddim_step_with_mean, ddim_step_with_logprob, ddim_prediction_with_logprob

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

@torch.no_grad()
def pipeline_using_smc(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
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
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

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

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        batch_size*batch_p, # num_images_per_prompt
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )
    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

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
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        
        # predict the noise residual
        noise_pred_tmp = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_tmp.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(
                noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
            )
        
        guidance = noise_pred - noise_pred_uncond if do_classifier_free_guidance else torch.zeros_like(noise_pred, device=noise_pred.device)

        return noise_pred, guidance, noise_pred_tmp

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
                    tmp_noise_pred, tmp_guidance, noise_pred_tmp = _pred_noise(tmp_latents, t)

                    tmp_pred_original_sample, _ = ddim_prediction_with_logprob(
                        self.scheduler, tmp_noise_pred, t, tmp_latents, **extra_step_kwargs
                    )

                    # Calculate rewards
                    decoded_image = _decode(tmp_pred_original_sample)
                    tmp_rewards = reward_fn(decoded_image).to(torch.float32)
                    tmp_log_twist_func = lookforward_fn(tmp_rewards).to(torch.float32) # 0.005 for aesthetic score

                    
                    # Calculate approximate guidance noise for maximizing reward
                    tmp_approx_guidance = torch.autograd.grad(outputs=tmp_log_twist_func, inputs=tmp_latents, grad_outputs=torch.ones_like(tmp_log_twist_func))[0].detach()
                    
                    # g = torch.autograd.grad(outputs=tmp_log_twist_func, inputs=decoded_image, grad_outputs=torch.ones_like(tmp_log_twist_func))[0].detach()
                    # g.max = 170
                    # g.min = -207
                    # g.mean = 0.001
                    # g.std = 3.07
                    
                    # g = torch.autograd.grad(outputs=tmp_log_twist_func, inputs=tmp_noise_pred, grad_outputs=torch.ones_like(tmp_log_twist_func))[0].detach()
                    # g.max 2798
                    # g,min -2186
                    # g.mean 0.1565
                    # g,std 141.25                  
                    
                    # g = torch.autograd.grad(outputs=tmp_log_twist_func, inputs=noise_pred_tmp, grad_outputs=torch.ones_like(tmp_log_twist_func))[0].detach()
                    # g.max 14000.
                    # g,min -11200.
                    # g.mean 0.0785
                    # g,std 639.5000
                    
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
                tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].detach().requires_grad_(True)
                with torch.no_grad():
                    tmp_latents = latents[batch_p*idx : batch_p*(idx+1)].detach()
                    tmp_noise_pred, tmp_guidance = _pred_noise(tmp_latents, t)

                noise_pred[batch_p*idx : batch_p*(idx+1)] = tmp_noise_pred.detach().clone()
                guidance[batch_p*idx : batch_p*(idx+1)] = tmp_guidance.detach().clone()

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
                    ################### Select Temperature ###################
                    if isinstance(tempering_schedule, float) or isinstance(tempering_schedule, int):
                        min_scale = torch.tensor([min((tempering_gamma * (i - start))**tempering_schedule, 1.)]*batch_size, device=device)
                        min_scale_next = torch.tensor([min(tempering_gamma * (i + 1 - start), 1.)]*batch_size, device=device)
                    elif tempering_schedule == "exp": # default setting
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
                    elif tempering == "schedule": # default setting
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
                    
                    ################### Weight & Resample (Importance Sampling) ###################

                    # Calculate weights for samples from proposal distribution
                    incremental_log_w = log_prob_diffusion + log_twist_func - log_prob_proposal - log_twist_func_prev # eq 14 of the paper
                    
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

                ################### Propose Particles ###################    
                # Sample from proposal distribution
                prev_sample, prev_sample_mean = ddim_step_with_mean(
                    self.scheduler, noise_pred, t, latents, **extra_step_kwargs
                )

                variance = get_variance(self.scheduler, t, prev_timestep)
                variance = eta**2 * _left_broadcast(variance, prev_sample.shape).to(device)
                std_dev_t = variance.sqrt()

                prop_latents = prev_sample + variance * approx_guidance ############# 
                manifold_deviation_trace = torch.cat([manifold_deviation_trace, ((variance * approx_guidance * (-noise_pred)).view(num_particles, -1).sum(dim=1).abs() / (noise_pred**2).view(num_particles, -1).sum(dim=1).sqrt()).unsqueeze(1)], dim=1)
                
                log_prob_diffusion = -0.5 * (prop_latents - prev_sample_mean).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                log_prob_diffusion = log_prob_diffusion.sum(dim=tuple(range(1, log_prob_diffusion.ndim)))
                log_prob_proposal = -0.5 * (prop_latents - prev_sample_mean - variance * approx_guidance).pow(2) / variance - torch.log(std_dev_t) - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                log_prob_proposal = log_prob_proposal.sum(dim=tuple(range(1, log_prob_proposal.ndim)))
                log_prob_diffusion[:] = torch.nan_to_num(log_prob_diffusion, nan=-1e6)
                log_prob_proposal[:] = torch.nan_to_num(log_prob_proposal, nan=1e6)

                log_prob_diffusion_trace = torch.cat([log_prob_diffusion_trace, (log_prob_diffusion_trace.transpose(0, 1)[-1] + log_prob_diffusion).unsqueeze(1)], dim=1) if i > 0 else log_prob_diffusion.unsqueeze(1)

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


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        guidance_rescale (`float`, *optional*, defaults to 0.7):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
    """
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

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

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents.to(prompt_embeds.dtype) if latents is not None else None,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob = ddim_step_with_logprob(
                self.scheduler, noise_pred, t, latents, **extra_step_kwargs
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image, has_nsfw_concept = self.run_safety_checker(
            image, device, prompt_embeds.dtype
        )
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(
        image, output_type=output_type, do_denormalize=do_denormalize
    )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    return image, has_nsfw_concept, all_latents, all_log_probs