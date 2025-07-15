import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import DoubleConv, Down, Up
from models.beamformer_heads import BeamformingHead

class FixedUNetBeamformer(nn.Module):
    def __init__(self, n_elements: int = 128, n_tasks: int = 3):
        super().__init__()
        self.n_elements = n_elements
        self.inc = DoubleConv(n_elements, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.das_head = BeamformingHead(64, n_elements)
        self.fdmas_head = BeamformingHead(64, n_elements)
        self.capon_head = BeamformingHead(64, n_elements)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        original_input = x
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        features = self.up3(x, x1)
        das_weights = self.das_head(features)
        fdmas_weights = self.fdmas_head(features)
        capon_weights = self.capon_head(features)
        original_input = torch.nan_to_num(original_input, nan=0.0, posinf=1.0, neginf=-1.0)
        das_output = torch.sum(das_weights * original_input, dim=1, keepdim=True)
        fdmas_output = torch.sum(fdmas_weights * original_input, dim=1, keepdim=True)
        capon_output = torch.sum(capon_weights * original_input, dim=1, keepdim=True)
        return {
            'das': das_output,
            'fdmas': fdmas_output,
            'capon': capon_output
        }
