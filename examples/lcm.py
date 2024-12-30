from diffusers import DiffusionPipeline, LatentConsistencyModelPipeline
from das.diffusers_patch.pipeline_using_SMC_LCM import pipeline_using_smc_lcm
import das.rewards as rewards
import torch
import numpy as np
from PIL import Image
import os

################### Configuration ###################
kl_coeff = 0.0001
n_steps = 8 # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_particles = 4
batch_p = 1
tempering_gamma = 0.1

# prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
prompt = "a black motorcycle is parked by the side of the road"
repeated_prompts = [prompt] * batch_p

# reward_fn = rewards.aesthetic_score(device = 'cuda')
reward_fn = rewards.PickScore(device = 'cuda')
image_reward_fn = lambda images: reward_fn(
                    images, 
                    repeated_prompts
                )

################### Initialize ###################
log_dir_lcm_smc = "logs/DAS_LCM/pick/qualitative"
os.makedirs(log_dir_lcm_smc, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to("cuda", torch.float32)
pipe.vae.to(dtype=torch.float32)
pipe.text_encoder.to(dtype=torch.float32)

################### Inference ###################
image = pipeline_using_smc_lcm(
    pipe,
    prompt=prompt,
    negative_prompt="",
    num_inference_steps=n_steps,
    eta=0.5,
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
image.save(f"{log_dir_lcm_smc}/{prompt} | reward: {reward}.png")
print(f"Saved in {log_dir_lcm_smc}/{prompt} | reward: {reward}.png")