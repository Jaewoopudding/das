from setuptools import setup, find_packages

setup(
    name="das",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numba==0.60.0",
        "numpy==2.0.0",
        "scipy==1.14.0",
        "matplotlib",
        "ml-collections==0.1.1",
        "absl-py==2.1.0",
        "diffusers==0.32.2",
        "accelerate==1.3.0",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "inflect==7.5.0",
        "pydantic==2.10.6",
        "transformers==4.48.2",
        "timm==1.0.14",
        "huggingface-hub==0.28.1",
        "clip",
        "lpips",
        "wandb",
    ]
)