import torch
from PIL import Image
import sys
import os
import copy
import gc
cwd = os.getcwd()
sys.path.append(cwd)

from tqdm import tqdm
import random
from collections import defaultdict
import prompts as prompts_file
import torch.distributed as dist
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
import datetime

from accelerate.logging import get_logger    
from accelerate import Accelerator
from absl import app, flags
from ml_collections import config_flags
import time

from online.model_utils import jpeg_compressibility, online_jpeg_loss_fn, generate_embeds_fn, evaluate_loss_fn, evaluate, generate_new_x, generate_new_x_smc, online_aesthetic_loss_fn
from online.dataset import D_explored

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/online.py:aesthetic", "Training configuration.")

from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)
    


def main(_):
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )


    if accelerator.is_main_process:
        wandb_args = {}
        wandb_args["name"] = config.run_name
        if config.debug:
            wandb_args.update({'mode':"disabled"})        
        accelerator.init_trackers(
            project_name="Online", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, config.run_name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    


    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    if config.pretrained.model.endswith(".safetensors") or config.pretrained.model.endswith(".ckpt"):
        pipeline = StableDiffusionPipeline.from_single_file(config.pretrained.model)
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)

    # freeze parameters of models to save more memory
    inference_dtype = torch.float32

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

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

    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
    
    embedding_fn = generate_embeds_fn(device = accelerator.device, torch_dtype = inference_dtype)    
    
    if config.reward_fn == "aesthetic":
        online_loss_fn = online_aesthetic_loss_fn(grad_scale=config.grad_scale,
                                    aesthetic_target=config.aesthetic_target,
                                    config=config,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
        eval_loss_fn = evaluate_loss_fn(grad_scale=config.grad_scale,
                                aesthetic_target=config.aesthetic_target,
                                accelerator = accelerator,
                                torch_dtype = inference_dtype,
                                device = accelerator.device)
    elif config.reward_fn == "jpeg_compressibility":
        online_loss_fn = online_jpeg_loss_fn(grad_scale=config.grad_scale,
                                    config=config,
                                    accelerator = accelerator,
                                    torch_dtype = inference_dtype,
                                    device = accelerator.device)
    
        eval_loss_fn = jpeg_compressibility(inference_dtype=inference_dtype,
                                    device = accelerator.device)
    else:
        raise NotImplementedError

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True 

    prompt_fn = getattr(prompts_file, config.prompt_fn)
    samping_prompt_fn = getattr(prompts_file, config.samping_prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
        )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size_per_gpu_available, 1, 1)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size_per_gpu_available, 1, 1)

    autocast = contextlib.nullcontext          
    #################### TRAINING ####################        

    num_fresh_samples = config.num_samples  # 64 samples take 4 minutes to generate
    assert len(num_fresh_samples) == config.train.num_outer_loop, "Number of outer loops must match the number of data counts"
    
    exp_dataset = D_explored(config, accelerator.device).to(accelerator.device, dtype=inference_dtype)
    exp_dataset.model = accelerator.prepare(exp_dataset.model)
    exp_dataset.base_model = accelerator.prepare(exp_dataset.base_model)

    global_step = 0
    for outer_loop in range(config.train.num_outer_loop):   
        ##### Generate a new sample x(i) ∼ p(i)(x) by running {p(i) and get a feedback y(i) =r(x(i)) + ε.
        num_new_x = num_fresh_samples[outer_loop]
        print(num_new_x)
        
        if outer_loop == 0 and not config.only_eval:
            if 'restore_initial_data_from' in config.keys():
                logger.info(f"Restore initial data from {config.restore_initial_data_from}")
                all_new_x = torch.load(config.restore_initial_data_from)
                all_new_x = all_new_x.to(accelerator.device)
            else:
                new_x, new_ims = generate_new_x(
                pipeline.unet, 
                num_new_x // config.train.num_gpus, 
                pipeline, 
                accelerator, 
                config, 
                inference_dtype, 
                samping_prompt_fn, 
                sample_neg_prompt_embeds, 
                embedding_fn)  
        
        else:
            os.makedirs(f"logs/{config.run_name}/eval_vis", exist_ok=True)
            reward_fn = lambda im_pix_un: online_loss_fn(im_pix_un, config, exp_dataset)[1]
            new_x, new_ims = generate_new_x_smc(
                _, 
                num_new_x // config.train.num_gpus, 
                pipeline, 
                accelerator, 
                config, 
                inference_dtype, 
                samping_prompt_fn, 
                sample_neg_prompt_embeds, 
                embedding_fn,
                reward_fn,
                outer_loop,
                eval_loss_fn
            )  
                 
        all_new_x = accelerator.gather(new_x)  # gather samples and distribute to all GPUs
        all_new_ims = accelerator.gather(new_ims)
        
        assert(len(all_new_x) == num_new_x), "Number of fresh online samples does not match the target number" 
        # if error occur, check config.train.num_gpus

        if not config.only_eval:
            ##### Construct a new dataset: D(i) = D(i−1) + (x(i), y(i))
            print(all_new_x.shape)
            if config.reward_fn == "aesthetic":
                exp_dataset.update(all_new_x)
            elif config.reward_fn == "jpeg_compressibility":
                _, all_new_y = eval_loss_fn(all_new_ims)
                exp_dataset.update(all_new_x, all_new_y.unsqueeze(1))
                del all_new_y
            del all_new_x
        
            # Train a pessimistic reward model r(x; D(i)) and a pessimistic bonus term g(i)(x; D(i))
            if config.train.optimism in ['none', 'UCB']:
                exp_dataset.train_MLP(accelerator, config, outer_loop=outer_loop)
            elif config.train.optimism == 'bootstrap':
                exp_dataset.train_bootstrap(accelerator, config, outer_loop=outer_loop)
            else:
                raise ValueError(f"Unknown optimism {config.train.optimism}")
        
            if accelerator.num_processes > 1:
                # sanity check model weight sync
                if config.train.optimism == 'bootstrap':
                    print(f"Process {accelerator.process_index} model 0 layer 0 bias: {exp_dataset.model.module.models[0].layers[0].bias.data}")
                else:
                    print(f"Process {accelerator.process_index} layer 0 bias: {exp_dataset.model.module.layers[0].bias.data}")
                print(f"Process {accelerator.process_index} x: {exp_dataset.x.shape}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    app.run(main)
