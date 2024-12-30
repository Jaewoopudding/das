from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
import torch
import numpy as np
import das.rewards as rewards
from PIL import Image
import os

################### Configuration ###################
kl_coeff = 0.0001
n_steps = 100
num_particles = 4
batch_p = 1
tempering_gamma = 0.008

prompt = "A close up of a handpalm with leaves growing from it."
repeated_prompts = [prompt] * batch_p

# reward_fn = rewards.aesthetic_score(device = 'cuda')
reward_fn = rewards.PickScore(device = 'cuda')
image_reward_fn = lambda images: reward_fn(
                    images, 
                    repeated_prompts
                )

################### Initialize ###################
log_dir_sdxl_smc = "logs/DAS_SDXL/pick/qualitative"
os.makedirs(log_dir_sdxl_smc, exist_ok=True)

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(n_steps)
pipe.to("cuda")
pipe.vae.to(dtype=torch.float32)
pipe.text_encoder.to(dtype=torch.float32)

################### Inference ###################
image = pipeline_using_smc_sdxl(
    pipe,
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=n_steps,
    output_type="pt",
    # SMC parameters
    num_particles=num_particles,
    batch_p=batch_p,
    tempering_gamma=tempering_gamma,
    reward_fn=image_reward_fn,
    kl_coeff=kl_coeff,
)[0]
reward = image_reward_fn(image).item()
image = (image[0].cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
image = Image.fromarray(image)
image.save(f"{log_dir_sdxl_smc}/{prompt} | reward: {reward}.png")
print(f"Saved in {log_dir_sdxl_smc}/{prompt} | reward: {reward}.png")