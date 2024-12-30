import copy
import torch
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

from aesthetic_scorer import AestheticScorerDiff, online_ScorerDiff, MLPDiff
from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import numpy as np
import torch.utils.checkpoint as checkpoint
import wandb
import contextlib
import torchvision
from transformers import AutoProcessor, AutoModel
import sys
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from das.diffusers_patch.pipeline_using_SMC import pipeline_using_smc

import datetime
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags

def jpeg_compressibility(inference_dtype=None, device=None):
    import io
    import numpy as np
    def loss_fn(im_pix_un):
        images = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        loss = torch.tensor(sizes, dtype=inference_dtype, device=device)
        rewards = -1 * loss

        return loss, rewards

    return loss_fn

def online_jpeg_loss_fn(grad_scale=0,
                     config=None,
                     device=None,
                     accelerator=None,
                     torch_dtype=None
                     ):
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = online_ScorerDiff(dtype=torch_dtype, config=config).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    
    for param in scorer.parameters():
        assert not param.requires_grad, "Scorer should not have any trainable parameters"

    target_size = 224
    def loss_fn(im_pix_un, config, D_exp):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        
        rewards = scorer(im_pix, config, D_exp)
        loss = -1 * rewards

        return loss * grad_scale, rewards
    return loss_fn
    

def online_aesthetic_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     config=None,
                     device=None,
                     accelerator=None,
                     torch_dtype=None
                     ):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])

    scorer = online_ScorerDiff(dtype=torch_dtype, config=config).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    scorer.eval()
    
    for param in scorer.parameters():
        assert not param.requires_grad, "Scorer should not have any trainable parameters"

    target_size = 224
    def loss_fn(im_pix_un, config, D_exp):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        
        rewards = scorer(im_pix, config, D_exp)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def evaluate_loss_fn(aesthetic_target=None,
                     grad_scale=0,
                     device=None,
                     accelerator=None,
                     torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def loss_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards,_ = scorer(im_pix)
        if aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - aesthetic_target)
        return loss * grad_scale, rewards
    return loss_fn

def evaluate(training_unet,latent,train_neg_prompt_embeds,prompts, pipeline, accelerator, inference_dtype, config, loss_fn):
    prompt_ids = pipeline.tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(accelerator.device)       
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
    
    all_rgbs_t = []
    for i, t in tqdm(
        enumerate(pipeline.scheduler.timesteps), 
        total=len(pipeline.scheduler.timesteps),
        disable=not accelerator.is_local_main_process
        ):
        t = torch.tensor([t],
                            dtype=inference_dtype,
                            device=latent.device)
        t = t.repeat(config.train.batch_size_per_gpu_available)

        noise_pred_uncond = training_unet(latent, t, train_neg_prompt_embeds).sample
        noise_pred_cond = training_unet(latent, t, prompt_embeds).sample
                
        grad = (noise_pred_cond - noise_pred_uncond)
        noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent, config.sample_eta).prev_sample
        
    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
    if "hps" in config.reward_fn:
        loss, rewards = loss_fn(ims, prompts)
    else:    
        _, rewards = loss_fn(ims)
    return ims, rewards

def generate_embeds_fn(device=None, torch_dtype=None): # get CLIP embeddings
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.requires_grad_(False)
    target_size = 224
    def embedding_fn(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(target_size, antialias=False)(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        embeds = scorer.generate_feats(im_pix)
        return embeds
    return embedding_fn

def generate_new_x(current_unet, 
            num_new_x, 
            pipeline, 
            accelerator, 
            config, 
            inference_dtype, 
            prompt_fn, 
            sample_neg_prompt_embeds, 
            embedding_fn):
    
    all_latents = torch.randn((num_new_x, 4, 64, 64), device=accelerator.device, dtype=inference_dtype) 

    if config.reward_fn == "aesthetic":
        good_prompt = "A stunning beautiful oil painting of a {prompt}, cinematic lighting, golden hour light."
    elif config.reward_fn == "jpeg_compressibility":
        good_prompt = "A {prompt} with smooth textures, minimal detail, and a clean, plain background."
    all_prompts, _ = zip(
        *[(good_prompt.format(prompt=prompt_fn()[0]), {}) 
            if random.random() < config.good_prompt_prop else prompt_fn() for _ in range(num_new_x)]
    )
    all_embeds = []
    all_ims = []
    
    with torch.no_grad():
        for index in tqdm(range(num_new_x // config.sample.batch_size_per_gpu_available),
                            total=num_new_x // config.sample.batch_size_per_gpu_available,
                            desc="Obtain fresh samples and feedbacks",
                            disable=not accelerator.is_local_main_process
                        ):
            latent = all_latents[config.sample.batch_size_per_gpu_available*index:config.sample.batch_size_per_gpu_available *(index+1)]
            prompts = all_prompts[config.sample.batch_size_per_gpu_available*index:config.sample.batch_size_per_gpu_available *(index+1)]
            
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)   
                
            pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
            
            for i, t in tqdm(
                enumerate(pipeline.scheduler.timesteps), 
                total=len(pipeline.scheduler.timesteps),
                disable=not accelerator.is_local_main_process
                ):
                t = torch.tensor([t],
                                    dtype=inference_dtype,
                                    device=latent.device)
                t = t.repeat(config.sample.batch_size_per_gpu_available)

                noise_pred_uncond = current_unet(latent, t, sample_neg_prompt_embeds).sample
                noise_pred_cond = current_unet(latent, t, prompt_embeds).sample
                        
                grad = (noise_pred_cond - noise_pred_uncond)
                noise_pred = noise_pred_uncond + config.sd_guidance_scale * grad
                
                latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent, config.sample_eta).prev_sample
            
            ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
            all_ims.append(ims)
            
            # for i in range(ims.shape[0]):
            #         eval_image = (ims[i,:,:,:].clone().detach() / 2 + 0.5).clamp(0, 1)
            #         pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            #         pil.save(f"./model/{i:03d}_{prompts[i]}.png")

            embeds = embedding_fn(ims)
            assert embeds.shape[0] == config.sample.batch_size_per_gpu_available
            assert embeds.shape[1] == 768
            all_embeds.append(embeds)
    return torch.cat(all_embeds, dim=0), torch.cat(all_ims, dim=0)

def generate_new_x_smc(current_unet, 
            num_new_x, 
            pipeline, 
            accelerator, 
            config, 
            inference_dtype, 
            prompt_fn, 
            sample_neg_prompt_embeds, 
            embedding_fn,
            reward_fn,
            outer_loop,
            eval_loss_fn
        ):

    num_prompts_per_gpu = 1
    batch_p = min(config.smc.num_particles, config.sample.batch_size_per_gpu_available)
    
    all_latents = torch.randn((config.smc.num_particles * num_new_x, 4, 64, 64), device=accelerator.device, dtype=inference_dtype) 

    if config.reward_fn == "aesthetic":
        good_prompt = "A stunning beautiful oil painting of a {prompt}, cinematic lighting, golden hour light."
    elif config.reward_fn == "jpeg_compressibility":
        good_prompt = "A {prompt} with smooth textures, minimal detail, and a clean, plain background."
    all_prompts, _ = zip(
        *[(good_prompt.format(prompt=prompt_fn()[0]), {}) 
            if random.random() < config.good_prompt_prop else prompt_fn() for _ in range(num_new_x)]
    )    
    all_embeds = []
    all_ims = []
    
    eval_image_vis = []

    with torch.no_grad():
        for index in tqdm(range(num_new_x // num_prompts_per_gpu),
                            total=num_new_x // num_prompts_per_gpu,
                            desc="Obtain fresh samples and feedbacks using SMC",
                            disable=not accelerator.is_local_main_process
                        ):
            latent = all_latents[config.smc.num_particles * num_prompts_per_gpu * index: config.smc.num_particles * num_prompts_per_gpu * (index+1)]
            prompts = all_prompts[num_prompts_per_gpu*index:num_prompts_per_gpu *(index+1)]
            repeated_prompts = [prompt for prompt in prompts for _ in range(batch_p)]
            
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)   
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            image_reward_fn = lambda images: reward_fn(
                images
            )

            sample_neg_prompt_embeds = sample_neg_prompt_embeds[0].repeat(num_prompts_per_gpu, 1, 1)       
            
            # Sample images
            print(batch_p)
            latent = pipeline_using_smc(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.steps,
                guidance_scale=config.sd_guidance_scale,
                eta=config.sample.eta,
                output_type="latent",
                latents=latent,
                num_particles=config.smc.num_particles,
                batch_p=batch_p,
                resample_strategy=config.smc.resample_strategy,
                ess_threshold=config.smc.ess_threshold,
                tempering=config.smc.tempering,
                tempering_schedule=config.smc.tempering_schedule,
                tempering_gamma=config.smc.tempering_gamma,
                tempering_start=config.smc.tempering_start,
                reward_fn=image_reward_fn,
                kl_coeff = config.train.kl_weight,
            )[0]
            
            ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
            all_ims.append(ims)

            embeds = embedding_fn(ims)
            assert embeds.shape[0] == num_prompts_per_gpu
            assert embeds.shape[1] == 768
            all_embeds.append(embeds)
            
            if False:
            # if not config.only_eval:
                if index < config.max_vis_images and accelerator.is_main_process:
                    image_un = ims[0,:,:,:].clone().detach()
                    eval_image = (image_un / 2 + 0.5).clamp(0, 1)
                    pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    reward = eval_loss_fn(image_un.unsqueeze(0))[1].item()

                    pil.save(f"logs/{config.run_name}/eval_vis/{outer_loop}_{index:03d}_{prompts[0]} | {reward:.2f}.png")
                    eval_image_vis.append(wandb.Image(pil, caption=f"{prompts[0]:.25} | {reward:.2f}")) 
            else:
                for i in range(ims.shape[0]):
                    image_un = ims[i,:,:,:].clone().detach()
                    eval_image = (image_un / 2 + 0.5).clamp(0, 1)
                    pil = Image.fromarray((eval_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    reward = eval_loss_fn(image_un.unsqueeze(0))[1].item()

                    pil.save(f"logs/{config.run_name}/eval_vis/{outer_loop}_{index:03d}_{prompts[0]} | {reward:.2f}.png")
                    eval_image_vis.append(wandb.Image(pil, caption=f"{prompts[0]:.25} | {reward:.2f}")) 

    accelerator.log({"eval_images": eval_image_vis},step=outer_loop)        

    return torch.cat(all_embeds, dim=0), torch.cat(all_ims, dim=0)

def prepare_pipeline(pipeline, accelerator, config, inference_dtype):
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    
    unet_1 = copy.deepcopy(pipeline.unet)
    unet_2 = copy.deepcopy(pipeline.unet)
    unet_3 = copy.deepcopy(pipeline.unet)
    unet_4 = copy.deepcopy(pipeline.unet)
    
    unet_1.requires_grad_(False)
    unet_2.requires_grad_(False)
    unet_3.requires_grad_(False)
    unet_4.requires_grad_(False)
    
    # disable safety checker
    pipeline.safety_checker = None    

    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )    

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(config.steps)
  
    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
    unet_1.to(accelerator.device, dtype=inference_dtype)
    unet_2.to(accelerator.device, dtype=inference_dtype)
    unet_3.to(accelerator.device, dtype=inference_dtype)
    unet_4.to(accelerator.device, dtype=inference_dtype)
    
    # Set correct lora layers
    lora_attn_procs_1 = {}
    for name in unet_1.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet_1.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet_1.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet_1.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet_1.config.block_out_channels[block_id]

        lora_attn_procs_1[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    
    lora_attn_procs_2 = copy.deepcopy(lora_attn_procs_1)
    lora_attn_procs_3 = copy.deepcopy(lora_attn_procs_1)
    lora_attn_procs_4 = copy.deepcopy(lora_attn_procs_1)
    
    unet_1.set_attn_processor(lora_attn_procs_1)
    unet_2.set_attn_processor(lora_attn_procs_2)
    unet_3.set_attn_processor(lora_attn_procs_3)
    unet_4.set_attn_processor(lora_attn_procs_4)
    

    class _Wrapper_1(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return unet_1(*args, **kwargs)

    unet_lora_1 = _Wrapper_1(unet_1.attn_processors)
    
    class _Wrapper_2(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return unet_2(*args, **kwargs)

    unet_lora_2 = _Wrapper_2(unet_2.attn_processors)
    
    class _Wrapper_3(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return unet_3(*args, **kwargs)

    unet_lora_3 = _Wrapper_3(unet_3.attn_processors)
    
    class _Wrapper_4(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return unet_4(*args, **kwargs)

    unet_lora_4 = _Wrapper_4(unet_4.attn_processors)
    
    return [pipeline.unet, unet_lora_1, unet_lora_2, unet_lora_3, unet_lora_4], [pipeline.unet, unet_1, unet_2, unet_3, unet_4]
    
    
    