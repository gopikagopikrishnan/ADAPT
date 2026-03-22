"""
model/adapt_model.py

FixedUNetBeamformer — encoder-decoder apodization network for ADAPT.

Input  : (B, N_ELEMENTS, H, W)  — ToFC RF tensor
Output : (B, N_ELEMENTS, H, W)  — softmax apodization weights (sum-to-1 per pixel)

Architecture:-
Encoder
  inc    DoubleConv(128 → 64)
  down1  MaxPool + DoubleConv(64 → 128)
  down2  MaxPool + DoubleConv(128 → 256)
  down3  MaxPool + DoubleConv(256 → 512)
Decoder
  up1    Up(768  → 256)  skip from down2
  up2    Up(384  → 128)  skip from down1
  up3    Up(192  →  64)  skip from inc
Head
  BeamformingHead(64 → N_ELEMENTS)  - channel-wise softmax
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation

class AntiRectifier(nn.Module):
    """Concatenate ReLU(x) and ReLU(-x) after channel-wise L2 normalisation.

    Preserves gradient signal for both polarities of raw RF values.
    Output channels = 2 × input channels.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


# Building blocks

class DoubleConv(nn.Module):
    """Conv-BN-AntiRectifier → Conv-BN-ReLU.

    The AntiRectifier doubles mid_channels, so the second convolution
    receives 2 * mid_channels as input.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels

        self.conv1 = nn.Conv2d(in_channels,       mid_channels,  3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.act1  = AntiRectifier()

        self.conv2 = nn.Conv2d(2 * mid_channels, out_channels,  3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.act2  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class Down(nn.Module):
    """MaxPool2d(2) + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample + skip-concat + DoubleConv.

    F.interpolate aligns the upsampled tensor to the skip-connection's
    spatial size, making the decoder robust to odd-shaped inputs.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        bilinear:     bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x2, x1], dim=1))


class BeamformingHead(nn.Module):
    """Task head: Conv → BN → ReLU → Conv → channel-wise softmax.

    Outputs softmax apodization weights that sum to 1 across elements
    at each spatial position: guarantees unity-gain beamforming.
    """

    def __init__(self, in_channels: int, n_elements: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,       n_elements * 2, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(n_elements * 2)
        self.conv2 = nn.Conv2d(n_elements * 2,    n_elements,     1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return F.softmax(self.conv2(x), dim=1)   # sum-to-1 over element axis


# Full model

class FixedUNetBeamformer(nn.Module):
    """U-Net apodization network for ADAPT.

    A single shared encoder-decoder outputs one set of element-wise
    apodization weights.  The head is task-specific (trained separately
    for DAS, FDMAS, and Capon targets).

    After training three copies, weight fusion is applied in
    inference/fuse_weights.py.
    """

    def __init__(self, n_elements: int = 128) -> None:
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(n_elements, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Decoder  (skip_ch + up_ch → out_ch)
        self.up1   = Up(768, 256, bilinear=True)   # 256 + 512
        self.up2   = Up(384, 128, bilinear=True)   # 128 + 256
        self.up3   = Up(192,  64, bilinear=True)   #  64 + 128

        # Task head — shared architecture, trained per task
        self.head  = BeamformingHead(64, n_elements)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        x : (B, N_ELEMENTS, H, W)

        Returns
        (B, N_ELEMENTS, H, W) — softmax apodization weights
        """
        x  = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        f  = self.up1(x4, x3)
        f  = self.up2(f,  x2)
        f  = self.up3(f,  x1)

        return self.head(f)
