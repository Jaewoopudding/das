from diffusers import DiffusionPipeline, DDIMScheduler
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc
import torch
import numpy as np
import das.rewards as rewards
from PIL import Image
import os

################### Configuration ###################
kl_coeff = 0.0001
n_steps = 100
num_particles = 4
batch_p = 2
tempering_gamma = 0.008

prompt = "cat and a dog"
repeated_prompts = [prompt] * batch_p

# reward_fn = rewards.aesthetic_score(device = 'cuda')
reward_fn = rewards.PickScore(device = 'cuda')
image_reward_fn = lambda images: reward_fn(
                    images, 
                    repeated_prompts
                )

################### Initialize ###################
log_dir_sd_smc = "logs/DAS_SD/pick/qualitative"
os.makedirs(log_dir_sd_smc, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(n_steps)
pipe.to("cuda", torch.float16)

################### Inference ###################
with torch.autocast('cuda'):
    image = pipeline_using_smc(
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
        verbose=True
    )[0]
reward = image_reward_fn(image).item()
image = (image[0].cpu().numpy() * 255).transpose(1, 2, 0).round().astype(np.uint8)
image = Image.fromarray(image)
image.save(f"{log_dir_sd_smc}/{prompt} | reward: {reward}.png")
print(f"Saved in {log_dir_sd_smc}/{prompt} | reward: {reward}.png")