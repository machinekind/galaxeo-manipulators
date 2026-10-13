"""The grasp student's network, rebuilt without torchvision.

    net = StudentNet()
    net.load_state_dict(ckpt["net"], strict=True)       # 130 entries
    a = net.eval()(img, obs)                            # img (B, 4, 120, 160) in [0, 1], obs normalised

Training builds `torchvision.models.resnet18` with a 4-channel first convolution, keeps
its children up to the average pool, and adds a proprioceptive MLP and a head. This file
reproduces exactly those modules with the same attribute names and `nn.Sequential`
indices, so the checkpoint's state dict loads strictly. The parameter-free slots of the
trunk (2 ReLU, 3 MaxPool, 8 AdaptiveAvgPool2d, 9 Flatten) stay in place for that reason.
"""

from __future__ import annotations

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                           if stride != 1 or cin != cout else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + idt)


def _layer(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))


def resnet18_trunk(in_channels: int = 4) -> nn.Sequential:
    """ResNet18 without its classifier: (B, in_channels, H, W) -> (B, 512)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
        _layer(64, 64, 1), _layer(64, 128, 2), _layer(128, 256, 2), _layer(256, 512, 2),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())


class StudentNet(nn.Module):
    """ResNet18 trunk on the 4-channel image, an MLP on the 19-vector, a tanh head."""

    def __init__(self, obs_dim: int = 19, act_dim: int = 5) -> None:
        super().__init__()
        self.trunk = resnet18_trunk(4)
        self.prop = nn.Sequential(nn.Linear(obs_dim, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512 + 128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, act_dim))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, img: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        # The heat channel (index 3) enters the trunk without the ImageNet normalisation.
        rgb = (img[:, :3] - self.mean) / self.std
        x = self.trunk(torch.cat([rgb, img[:, 3:]], 1))
        return torch.tanh(self.head(torch.cat([x, self.prop(obs)], -1)))
