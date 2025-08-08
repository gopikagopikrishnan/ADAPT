import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import DoubleConv, Down, Up
from model.beamformer_heads import BeamformingHead

class FixedUNetBeamformer(nn.Module):
    def __init__(self, n_elements=128, n_tasks=3):
        super().__init__()
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

    def forward(self, x):
        original_input = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        features = self.up3(x, x1)
        #das = torch.sum(self.das_head(features) * original_input, dim=1, keepdim=True)
        #fdmas = torch.sum(self.fdmas_head(features) * original_input, dim=1, keepdim=True)
        capon = torch.sum(self.capon_head(features) * original_input, dim=1, keepdim=True)
        # return {'das': das, 'fdmas': fdmas, 'capon': capon}
        return capon
