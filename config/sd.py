import ml_collections
import imp
import os
from config.general import general

def smc():
    config = general()
    config.project_name = "DAS_SD"
    config.smc = ml_collections.ConfigDict()

    config.smc.num_particles = 4
    config.smc.resample_strategy = "ssp"
    config.smc.ess_threshold = 0.5
    
    config.smc.tempering = "schedule" # either adaptive, FreeDoM, schedule or None
    config.smc.tempering_schedule = "exp" # either float(exponent of polynomial), "exp", or "adaptive"
    config.smc.tempering_gamma = 0.008
    config.smc.tempering_start = 0

    config.smc.verbose = True

    config.sample.num_steps = 100
    config.sample.eta = 1.

    config.sample.batch_size = 2
    config.max_vis_images = 2

    return config

def aesthetic():
    config = smc()
    config.reward_fn = "aesthetic"
    config.prompt_fn = "eval_simple_animals"

    config.smc.kl_coeff = 0.005

    return config

def clip():
    print("CLIP Score")
    config = smc()
    config.reward_fn = "clip"
    config.prompt_fn = "eval_hps_v2_all"
    
    config.smc.kl_coeff = 0.01

    return config

def multi():
    print("Aesthetic + CLIP Score")
    config = smc()
    config.reward_fn = "multi"
    config.prompt_fn = "eval_hps_v2_all"

    config.aes_weight = 1.0
    
    config.smc.kl_coeff = 0.005

    return config

def pick():
    print("PickScore")
    config = smc()
    config.reward_fn = "pick"
    config.prompt_fn = "eval_hps_v2_all"
    
    config.smc.kl_coeff = 0.0001

    return config

def get_config(name):
    return globals()[name]()
