"""Resnet implementation in pytorch

See Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385v1
"""

import torch
from torch import nn

class BasicBlock(nn.Module):
    """Resnet Block for resnet 18 and resnet 34."""

    # if basic block and bottleneck block have different output size
    # we use expansion to distinct.    
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        """Resnet block for building resnet.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            stride: The stride.
        """
        super().__init__()
        self._residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        if stride != 2 or in_channels != BasicBlock.expansion * out_channels:
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            self._shortcut = nn.Sequential()

    def forward(self, x):
        return nn.ReLU(inplace=True)(self._residual_function(x) + self._shortcut(x))


class BottleNeck(nn.Module):
    """Resnet block for resnet over 50 layers."""

    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        """Resnet block for building resnet.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            stride: The stride.
            batch_norm: If ture, batch norm layer will be added between conv2d.
        """
        super().__init__()
        self._residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, bias=False, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )
        else:
            self._shortcut = nn.Sequential()
        self._stride = stride

    def forward(self, x):
        return nn.ReLU(inplace=True)(self._residual_function(x) + self._shortcut(x))


class ResNet(nn.Module):
    """Bulding RestNet layers."""

    def __init__(self, config, batch_norm: bool=True, num_classes: int=100):
        """Constructor for ResNet class."""
        super().__init__()
        self._config = config
        self._num_classes = num_classes
        self._name = config["name"]

        if self._config["block"] == "BasicBlock":
            self._block = BasicBlock
        elif self._config["block"] == "BottleNeck":
            self._block = BottleNeck
        else:
            raise ValueError(f"Block not found for {self._config['block']}")

        self._blocks = config["blocks"]
        # Why 4 ??
        assert len(self._blocks) == 4
        self._in_channels = 64
        self._conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self._conv2_res = self._make_layer(64, self._blocks[0], 1)
        self._conv3_res = self._make_layer(128, self._blocks[1], 2)
        self._conv4_res = self._make_layer(256, self._blocks[2], 2)
        self._conv5_res = self._make_layer(512, self._blocks[3], 2)
        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 512 is for 3x32x32 input
        self._fc = nn.Linear(512 * self._block.expansion, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Module:
        """Make resnet layers, one layer contains more than one residual blocks.

        Args:
            out_channels: Number of output channels for this layer.
            num_blocks: Number of residual blocks.
            stride: The stide of the first block of this layer.

        Return:
            Return a resnet layer.
        """
        # The first block could be 1 or 2, other blocks need to be 1.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self._block(self._in_channels, out_channels, stride))
            self._in_channels = out_channels * self._block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self._conv1(x)
        output = self._conv2_res(output)
        output = self._conv3_res(output)
        output = self._conv4_res(output)
        output = self._conv5_res(output)
        output = self._avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self._fc(output)
        return output

    def name(self):
        return self._name

    def input_shape(self):
        """Input shape for model."""
        # TODO(weich): Hardcoded for cifar100 dataset.
        return [3, 32, 32]


def resnet18_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return ResNet(
        config={
            "name": "resnet18",
            "block": "BasicBlock",
            "blocks": [2, 2, 2, 2],
        },
        num_classes=num_classes,
    )


def resnet34_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return ResNet(
        config={
            "name": "resnet34",
            "block": "BasicBlock",
            "blocks": [3, 4, 6, 3],
        },
        num_classes=num_classes,
    )


def resnet50_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return ResNet(
        config={
            "name": "resnet50",
            "block": "BottleNeck",
            "blocks": [3, 4, 6, 3],
        },
        num_classes=num_classes,
    )


def resnet101_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return ResNet(
        config={
            "name": "resnet101",
            "block": "BottleNeck",
            "blocks": [3, 4, 23, 3],
        },
        num_classes=num_classes,
    )


def resnet152_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return ResNet(
        config={
            "name": "resnet152",
            "block": "BottleNeck",
            "blocks": [3, 8, 36, 3],
        },
        num_classes=num_classes,
    )
