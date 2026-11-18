"""Li et al. (2024) Figure 3 Model B: Model A scalar + 11 physical inputs.

Place next to the original `model_loader.py`. Original Model A files are unchanged.
This module defines a randomly initialized, end-to-end trainable Model B.
"""
import torch
from torch import nn
from model_loader import ModelA


class ModelB(nn.Module):
    def __init__(self, num_physics=11):
        super().__init__()
        if num_physics != 11:
            raise ValueError('Paper Model B requires 11 physical inputs')
        self.model_a = ModelA()
        self.fusion = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.fusion.apply(ModelA.initialize)

    def forward(self, images, physics):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 256, 256):
            raise ValueError('images must have shape [batch, 1, 256, 256]')
        if physics.ndim != 2 or physics.shape[1] != 11:
            raise ValueError('physics must have shape [batch, 11]')
        if images.shape[0] != physics.shape[0]:
            raise ValueError('images and physics must have the same batch size')
        a_output = self.model_a(images).unsqueeze(1)  # [batch] -> [batch, 1]
        combined = torch.cat((a_output, physics), dim=1)  # [batch, 12]
        return self.fusion(combined).squeeze(1)  # [batch]
