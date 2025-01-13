
<!-- TITLE -->
# Feedback Efficient Online Black-Box Optimization using Diffusion Models

We combined DAS with [SEIKO](https://github.com/zhaoyl18/SEIKO) to solve online black-box optimization feedback efficiently. Instead of fine-tuning diffusion models using surrogate models as in SEIKO, we use DAS to generate samples that maximize the surrogate model to avoid over-optimization and enable more frequent update of the surrogate model.

## Usage

To use DAS for online black-box optimization with aesthetic score or jpeg compressibility as black-box rewards:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/UCB_smc.py:aesthetic
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/Bootstrap_smc.py:aesthetic

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/UCB_smc.py:jpeg
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/Bootstrap_smc.py:jpeg
```

The above codes save trained surrogate reward models. To generate samples, change config.reward_model_path to the final surrogate model checkpoint and run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/UCB_smc.py:evaluate
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/Bootstrap_smc.py:evaluate
```

### Acknowledgement

This codebase is directly built on top of [SEIKO](https://github.com/zhaoyl18/SEIKO). We are thankful to the authors for providing the codebases.