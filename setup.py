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
        "diffusers==0.31.0",
        "accelerate==0.17",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.30.2",
        "huggingface-hub==0.23.4",
        "wandb",
    ],
)