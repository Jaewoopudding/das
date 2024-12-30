from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import torchvision

ASSETS_PATH = resources.files("assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, dtype=self.dtype)
        self.mlp = MLP().to(self.device, dtype=self.dtype)

        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=self.device)
        self.mlp.load_state_dict(state_dict)

        self.target_size =  224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])

        self.eval()

    def __call__(self, images):
        inputs = torchvision.transforms.Resize(self.target_size)(images)
        inputs = self.normalize(inputs).to(self.dtype)
        embed = self.clip.get_image_features(pixel_values=inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        return self.mlp(embed).squeeze(1)