"""
model.py
UNet-based apodization network for deep-learning beamforming.

Architecture
Input  : (B, C, H, W)  — Time-of-Flight Corrected (ToFC) RF data, C = N_ELEMENTS (128 channels)
Output : tuple of softmax apodization maps from head(B, C, H, W)
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation
class AntiRectifier(nn.Module):
    """Concatenates F.relu(x) and F.relu(-x) after L2 normalisation.

    Enables gradient flow for both positive and negative raw-RF values.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = F.normalize(x, p=2, dim=1, eps=self.eps)
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


# Blocks

class DoubleConv(nn.Module):
    """Two sequential Conv-BN-Activation blocks.

    The first activation is ``AntiRectifier`` (doubles channels), so the
    second convolution takes ``2 * mid_channels`` as input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.act1  = AntiRectifier()

        # AntiRectifier doubles the channel dimension
        self.conv2 = nn.Conv2d(2 * mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.act2  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class Down(nn.Module):
    """MaxPool2d(2) followed by a DoubleConv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsampling (or transposed convolution) + DoubleConv.

    Spatial dimensions of the skip-connection are matched via F.interpolate before concatenation, making the decoder robust to odd-sized inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x2, x1], dim=1))


class BeamformingHead(nn.Module):
    """Task-specific head that produces a softmax apodization map."""

    def __init__(self, in_channels: int, n_elements: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_elements * 2, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(n_elements * 2)
        self.conv2 = nn.Conv2d(n_elements * 2, n_elements, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return F.softmax(self.conv2(x), dim=1)


# Model call

class FixedUNetBeamformer(nn.Module):
    """Encoder-decoder beamformer with task-specific.

    Encoder
    -------
    inc   : DoubleConv(n_elements → 64)
    down1 : MaxPool + DoubleConv(64 → 128)
    down2 : MaxPool + DoubleConv(128 → 256)
    down3 : MaxPool + DoubleConv(256 → 512)

    Decoder
    -------
    up1   : Up(768  → 256)   — skip from down2 (256) + up-sampled down3 (512)
    up2   : Up(384  → 128)   — skip from down1 (128) + up-sampled up1 (256)
    up3   : Up(192  → 64)    — skip from inc   (64)  + up-sampled up2 (128)

    Heads
    -----
    head : BeamformingHead(64 → n_elements)

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        (beamforming_weights), (B, n_elements, H, W)
    """

    def __init__(self, n_elements: int = 128) -> None:
        super().__init__()
        self.n_elements = n_elements

        # Encoder
        self.inc   = DoubleConv(n_elements, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Decoder  (in_channels = skip + up-sampled)
        self.up1 = Up(768, 256, bilinear=True)   # 512 + 256
        self.up2 = Up(384, 128, bilinear=True)   # 256 + 128
        self.up3 = Up(192,  64, bilinear=True)   # 128 +  64

        # Task head
        # self.das_head   = BeamformingHead(64, n_elements)
        # self.fdmas_head = BeamformingHead(64, n_elements)
        self.head = BeamformingHead(64, n_elements)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        f = self.up1(x4, x3)
        f = self.up2(f, x2)
        f = self.up3(f, x1)

        return self.head(f)
