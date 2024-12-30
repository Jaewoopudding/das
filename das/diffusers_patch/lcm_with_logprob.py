import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name\

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas

def lcm_step_with_mean(
    self,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
    Returns:
        [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )
    if self.step_index is None:
        self._init_step_index(timestep)

    # 1. get previous step value
    prev_step_index = self.step_index + 1
    if prev_step_index < len(self.timesteps):
        prev_timestep = self.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 3. Get scalings for boundary conditions
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

    # 4. Compute the predicted original sample x_0 based on the model parameterization
    if self.config.prediction_type == "epsilon":  # noise-prediction
        predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == "sample":  # x-prediction
        predicted_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":  # v-prediction
        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for `LCMScheduler`."
        )

    # 5. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        predicted_original_sample = self._threshold_sample(predicted_original_sample)
    elif self.config.clip_sample:
        predicted_original_sample = predicted_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 6. Denoise model output using boundary conditions
    denoised = c_out * predicted_original_sample + c_skip * sample

    # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    # Noise is not used on the final timestep of the timestep schedule.
    # This also means that noise is not used for one-step sampling.
    if self.step_index != self.num_inference_steps - 1:
        noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype
        )
        prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
    else:
        prev_sample = denoised
    # upon completion increase step index by one
    self._step_index += 1

    return prev_sample, alpha_prod_t_prev.sqrt() * denoised

def lcm_prediction(
    self,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
    Returns:
        [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )
    if self.step_index is None:
        self._init_step_index(timestep)

    # 1. get previous step value
    prev_step_index = self.step_index + 1
    if prev_step_index < len(self.timesteps):
        prev_timestep = self.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    # 2. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    # 3. Get scalings for boundary conditions
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

    # 4. Compute the predicted original sample x_0 based on the model parameterization
    if self.config.prediction_type == "epsilon":  # noise-prediction
        predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == "sample":  # x-prediction
        predicted_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":  # v-prediction
        predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction` for `LCMScheduler`."
        )

    # 5. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        predicted_original_sample = self._threshold_sample(predicted_original_sample)
    elif self.config.clip_sample:
        predicted_original_sample = predicted_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 6. Denoise model output using boundary conditions
    denoised = c_out * predicted_original_sample + c_skip * sample

    # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    # Noise is not used on the final timestep of the timestep schedule.
    # This also means that noise is not used for one-step sampling.
    if self.step_index != self.num_inference_steps - 1:
        noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype
        )
        prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
    else:
        prev_sample = denoised

    return denoised.type(sample.dtype)