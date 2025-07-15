import torch
import torch.nn as nn
import torch.nn.functional as F

class BeamformingHead(nn.Module):
    def __init__(self, in_channels: int, n_elements: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, n_elements * 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_elements * 2)
        self.conv2 = nn.Conv2d(n_elements * 2, n_elements, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        weights = self.conv2(x)
        return F.softmax(weights, dim=1)
