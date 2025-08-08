import torch
import torch.nn as nn
import torch.nn.functional as F

class AntiRectifier(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)
