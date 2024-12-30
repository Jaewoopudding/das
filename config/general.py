import ml_collections
import os



def general():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.project_name = "DiffusionSampleBaseline"
    config.max_vis_images = 2
    config.run_name = ""
    
    # prompting
    config.prompt_fn = "simple_animals"
    config.reward_fn = "aesthetic"
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "fp16"
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 42    

    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 50
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 1

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # revision of the model to load.
    pretrained.revision = "main"

    return config

def get_config(name):
    return globals()[name]()