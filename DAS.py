from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc, pipeline_with_logprob
from das.diffusers_patch.pipeline_using_SMC_SDXL import pipeline_using_smc_sdxl
from das.diffusers_patch.pipeline_using_SMC_LCM import pipeline_using_smc_lcm
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from DiffusionSampler import DiffusionModelSampler
import matplotlib.pyplot as plt

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

class DAS(DiffusionModelSampler):

    def __init__(self, config):
        super().__init__(config)

        if "xl" in self.config.pretrained.model:
            print("Using SDXL")
            self.pipeline_using_smc = pipeline_using_smc_sdxl
        elif "lcm" in self.config.pretrained.model or "LCM" in self.config.pretrained.model:
            print("Using LCM")
            self.pipeline_using_smc = pipeline_using_smc_lcm
        else:
            print("Using SD")
            self.pipeline_using_smc = pipeline_using_smc
            self.eval_images_pipeline = pipeline_with_logprob
        self.negative_log_dir = f"logs/{self.config.project_name}/{self.config.reward_fn}/{self.config.run_name}/eval_neg"
        os.makedirs(self.negative_log_dir, exist_ok=True)

    def sample_images(self, train=False):
        """Sample images using the diffusion model."""
        
        samples = []

        num_prompts_per_gpu = 1 if self.config.smc.num_particles >= self.config.sample.batch_size else int(self.config.sample.batch_size / self.config.smc.num_particles)
        batch_p = min(self.config.smc.num_particles, self.config.sample.batch_size)

        # Generate prompts and latents
        prompts, prompt_metadata = self.eval_prompts, self.eval_prompt_metadata

        latents_0 = torch.randn(
            (self.config.smc.num_particles*self.config.max_vis_images, self.pipeline.unet.config.in_channels, self.pipeline.unet.sample_size, self.pipeline.unet.sample_size),
            device=self.accelerator.device,
            dtype=self.inference_dtype,
        )

        with torch.no_grad():
            for vis_idx in tqdm(
                range(self.config.max_vis_images//num_prompts_per_gpu),
                desc=f"Sampling images",
                disable=not self.accelerator.is_local_main_process,
                position=0,
            ):
                prompts_batch = prompts[vis_idx*num_prompts_per_gpu : (vis_idx+1)*num_prompts_per_gpu]
                repeated_prompts = [prompt for prompt in prompts_batch for _ in range(batch_p)]
                
                latents_batch = latents_0[vis_idx*self.config.smc.num_particles*num_prompts_per_gpu : (vis_idx+1)*self.config.smc.num_particles*num_prompts_per_gpu]
    
                # convert reward function to get image as only input
                image_reward_fn = lambda images: self.reward_fn(
                    images, 
                    repeated_prompts
                )

                # Encode prompts
                prompt_ids = self.pipeline.tokenizer(
                    prompts_batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.pipeline.tokenizer.model_max_length,
                ).input_ids.to(self.accelerator.device)
                prompt_embeds = self.pipeline.text_encoder(prompt_ids)[0]
                
                # Sample images
                with self.autocast():
                    images, log_w, normalized_w, latents, \
                    all_log_w, resample_indices, ess_trace, \
                    scale_factor_trace, rewards_trace, manifold_deviation_trace, log_prob_diffusion_trace \
                    = self.pipeline_using_smc(
                        self.pipeline,
                        prompt=list(prompts_batch),
                        negative_prompt=[""]*len(prompts_batch),
                        num_inference_steps=self.config.sample.num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        eta=self.config.sample.eta,
                        output_type="pt",
                        latents=latents_batch,
                        num_particles=self.config.smc.num_particles,
                        batch_p=batch_p,
                        resample_strategy=self.config.smc.resample_strategy,
                        ess_threshold=self.config.smc.ess_threshold,
                        tempering=self.config.smc.tempering,
                        tempering_schedule=self.config.smc.tempering_schedule,
                        tempering_gamma=self.config.smc.tempering_gamma,
                        tempering_start=self.config.smc.tempering_start,
                        reward_fn=image_reward_fn,
                        kl_coeff=self.config.smc.kl_coeff,
                        verbose=self.config.smc.verbose
                    )
                    
                    negative_images, _, _, _ = pipeline_with_logprob(
                        self.pipeline,
                        prompt=list(prompts_batch),
                        num_inference_steps=self.config.sample.num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        negative_prompt=[""]*len(prompts_batch),
                        eta=self.config.sample.eta,
                        output_type="pt",
                    )
                    
                    
                self.info_eval_vis["eval_ess"].append(ess_trace)
                self.info_eval_vis["scale_factor_trace"].append(scale_factor_trace)
                self.info_eval_vis["rewards_trace"].append(rewards_trace)
                self.info_eval_vis["manifold_deviation_trace"].append(manifold_deviation_trace)
                self.info_eval_vis["log_prob_diffusion_trace"].append(log_prob_diffusion_trace)

                rewards = self.reward_fn(images, prompts_batch)
                negative_sample_rewards = self.reward_fn(negative_images, prompts_batch)
                
                self.info_eval_vis["eval_rewards_img"].append(rewards.clone().detach())
                self.info_eval_vis["eval_negative_rewards_img"].append(negative_sample_rewards.clone().detach())
                self.info_eval_vis["eval_image"].append(images.clone().detach())
                self.info_eval_vis["eval_prompts"] = list(self.info_eval_vis["eval_prompts"]) + list(prompts_batch)
                self.info_eval_vis["eval_negative_image"].append(negative_images.clone().detach())

    def log_evaluation(self, epoch=None, inner_epoch=None):
        self.info_eval = {k: torch.mean(torch.stack(v)) for k, v in self.info_eval.items()}
        self.info_eval = self.accelerator.reduce(self.info_eval, reduction="mean")

        ims = torch.cat(self.info_eval_vis["eval_image"])
        negative_ims = torch.cat(self.info_eval_vis["eval_negative_image"])
        rewards = torch.cat(self.info_eval_vis["eval_rewards_img"])
        negative_rewards = torch.cat(self.info_eval_vis["eval_negative_rewards_img"])
        prompts = self.info_eval_vis["eval_prompts"]
        
        self.info_eval["eval_rewards"] = rewards.mean()
        self.info_eval["eval_rewards_std"] = rewards.std()

        self.accelerator.log(self.info_eval, step=self.global_step)

        images  = []
        for i, image in enumerate(ims):
            prompt = prompts[i]
            reward = rewards[i]
            
            negative_image = negative_ims[i]
            negative_reward = negative_rewards[i]
                
            if image.min() < 0: # normalize unnormalized images
                image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                
            if negative_image.min() < 0: # normalize unnormalized images
                negative_image = (negative_image.clone().detach() / 2 + 0.5).clamp(0, 1)

            pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil_negative = Image.fromarray((negative_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            if self.config.reward_fn == "inpaint":
                if epoch is not None and inner_epoch is not None:
                    caption = f"{epoch:03d}_{inner_epoch:03d}_{self.config.inpaint.sample_name:.25} | {reward:.2f}"
                else:
                    caption = f"{self.config.inpaint.sample_name:.25} | {reward:.2f}"
                pil_target = Image.fromarray((self.masked_target[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil_target.save(f"{self.log_dir}/masked {self.config.inpaint.sample_name}_{self.config.inpaint.x}_{self.config.inpaint.x+self.config.inpaint.width}_{self.config.inpaint.y}_{self.config.inpaint.y+self.config.inpaint.height}.png")
            else: 
                if epoch is not None and inner_epoch is not None:
                    caption = f"{epoch:03d}_{inner_epoch:03d}_{i:03d}_{prompt} | reward: {reward}"
                    negative_caption = f"{epoch:03d}_{inner_epoch:03d}_{i:03d}_{prompt} | reward: {negative_reward}"
                else:
                    caption = f"{i:03d}_{prompt} | reward: {reward}"
                    negative_caption = f"{i:03d}_{prompt} | reward: {negative_reward}"
            pil.save(f"{self.log_dir}/{caption}.png")
            pil_negative.save(f"{self.negative_log_dir}/{negative_caption}.png")

            pil = pil.resize((256, 256))
            if self.config.reward_fn == "inpaint":
                caption = f"{self.config.inpaint.sample_name:.25} | {reward:.2f}"
            else: 
                caption = f"{prompt:.25} | {reward:.2f}"
            images.append(wandb.Image(pil, caption=caption)) 

        self.accelerator.log({"eval_images": images},step=self.global_step)

        # Log additional details if needed
        self.logger.info(f"Logged Evaluation results for step {self.global_step}")

        rewards = torch.cat(self.info_eval_vis["eval_rewards_img"])
        prompts = self.info_eval_vis["eval_prompts"]

        ess_trace = torch.cat(self.info_eval_vis["eval_ess"])
        scale_factor_trace = torch.cat(self.info_eval_vis["scale_factor_trace"])
        rewards_trace = torch.cat(self.info_eval_vis["rewards_trace"])
        manifold_deviation_trace = torch.cat(self.info_eval_vis["manifold_deviation_trace"])
        log_prob_diffusion_trace = torch.cat(self.info_eval_vis["log_prob_diffusion_trace"])
        
        for i, ess in enumerate(ess_trace):

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            ax1.plot(range(len(ess)), ess, 'b-')
            caption = f"{i:03d}_{prompts[i]} | reward: {rewards[i]}"
            os.makedirs(f"{self.log_dir}/{caption}", exist_ok=True)

            plt.savefig(f"{self.log_dir}/{caption}/ess.png")
            plt.clf()

            plt.plot(rewards_trace[i])
            plt.savefig(f"{self.log_dir}/{caption}/intermediate_rewards.png")
            plt.clf()

            plt.plot(manifold_deviation_trace[i])
            plt.savefig(f"{self.log_dir}/{caption}/manifold_deviation.png")
            plt.clf()

            plt.plot(log_prob_diffusion_trace[i])
            plt.savefig(f"{self.log_dir}/{caption}/log_prob_diffusion.png")
            plt.clf()

            np.save(f"{self.log_dir}/{caption}/ess.npy", ess)
            np.save(f"{self.log_dir}/{caption}/manifold_deviation.npy", manifold_deviation_trace[i])
            np.save(f"{self.log_dir}/{caption}/log_prob_diffusion.npy", log_prob_diffusion_trace[i])




FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/sd.py", "Sampling configuration.")

def main(_):
    # Load the configuration
    config = FLAGS.config

    # Initialize the trainer with the configuration
    sampler = DAS(config)

    # Run sampling
    sampler.run_evaluation()

if __name__ == "__main__":
    app.run(main)
