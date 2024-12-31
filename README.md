# DAS (Diffusion Alignment as Sampling)

This project implements Diffusion Alignment as Sampling (DAS)

## Installation

```bash
conda create -n das python=3.10
conda activate das
pip install -e .
```

Install hpsv2 from [HPSv2](https://github.com/tgxs002/HPSv2). Recommend using method 2 (installing locally) to avoid errors.

## Usage

### Single prompt

DAS is implemented over diffusers library, making it easy to use. Minimal code for usage with single test prompt can be found in the examples folder.

```bash
python examples/sd.py
python examples/sdxl.py
python examples/lcm.py
```

### Multiple prompts with Multiple gpus

To run Aesthetic score experiment with Stable Diffusion 1.5:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/sd.py:aesthetic
```

To run PickScore experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/sd.py:pick
```

To run multi-objective (Aesthetic score + CLIPScore) experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/sd.py:multi
```
where the ratio of two rewards can be customized in the config file.

Similarly, to use [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) or [LCM](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/sdxl.py:aesthetic
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/sdxl.py:pick
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/sdxl.py:multi

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/lcm.py:aesthetic
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/lcm.py:pick
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch DAS.py --config config/lcm.py:multi
```

## Evaluation

Evaluation for cross-reward generalization and sample diversity can be performed using the `eval.ipynb` Jupyter notebook. 

## Online black-box optimization

Online black-box optimization experiments can be conducted in SEIKO folder which use codes from the [SEIKO](https://github.com/zhaoyl18/SEIKO) repository. To use DAS for online black-box optimization with aesthetic score or jpeg compressibility as black-box rewards:

```bash
cd SEIKO

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/UCB_smc.py:aesthetic
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/Bootstrap_smc.py:aesthetic

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/UCB_smc.py:jpeg
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/Bootstrap_smc.py:jpeg
```

The above codes save trained surrogate reward models. To generate samples using the surrogate reward, change config.reward_model_path and run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/UCB_smc.py:evaluate
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch online/online_main_smc.py --config config/Bootstrap_smc.py:evaluate
```

## Toy Examples

The Mixture of Gaussians ans Swiss roll experiments can be reproduced using Jupyter notebooks in the notebooks folder.

## Acknowledgments

We sincerely thank those who have open-sourced their works including, but not limited to, the repositories below:
- https://github.com/huggingface/diffusers
- https://github.com/kvablack/ddpo-pytorch
- https://github.com/mihirp1998/AlignProp
- https://github.com/zhaoyl18/SEIKO
- https://github.com/DPS2022/diffusion-posterior-sampling
- https://github.com/vvictoryuki/FreeDoM/tree/main
- https://github.com/KellyYutongHe/mpgd_pytorch
- https://github.com/blt2114/twisted_diffusion_sampler
- https://github.com/nchopin/particles
