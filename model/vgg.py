"""vgg implementation in pytorch.

See Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""

import torch
from torch import nn


VGG_CONFIG = {
    "vgg11": [64,     "M", 128,      "M", 256, 256,           "M", 512, 512,           "M", 512, 512,           "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256,           "M", 512, 512,           "M", 512, 512,           "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256,      "M", 512, 512, 512,      "M", 512, 512, 512,      "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):

    def __init__(self, config, batch_norm: bool=True, num_classes: int=100):
        """
        Note: This model only supports cifar100 datasets.
        TODO: Support multiple datasets.

        Args:
            config (object): Model configuration struct.
            batch_norm (bool): Whether or not to use batch_norm after conv2d.
            nm_classse (int): number of classes for classification.
        """
        super().__init__()
        self._name = config["name"]
        self._extractor = _make_layers(config["params"], batch_norm)
        self._classfier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        output = self._extractor(x)
        output = output.view(output.size()[0], -1)
        output = self._classfier(output)
        return output

    def name(self):
        return self._name

    def input_shape(self):
        """Input shape for model."""
        # TODO(weich): Hardcoded for cifar100 dataset.
        return [3, 32, 32]


def _make_layers(config, batch_norm: bool=True) -> nn.Sequential:
    """
    Args:
        config (list): List defines network structure for VGG variances.
        batch_norm (bool): Whether or not to use batch_norm after each Conv2d.
    """
    layers = []
    input_channels = 3

    for layer in config:
        if layer == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        assert type(layer) == int
        layers.append(nn.Conv2d(input_channels, layer, kernel_size=3, padding=1))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(layer))

        layers.append(nn.ReLU(inplace=True))
        input_channels = layer

    
    return nn.Sequential(*layers)

def vgg11_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return VGG(
        config={
            "name": "vgg11",
            "params": VGG_CONFIG["vgg11"],
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )

def vgg13_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return VGG(
        config={
            "name": "vgg13",
            "params": VGG_CONFIG["vgg13"],
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )

def vgg16_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return VGG(
        config={
            "name": "vgg16",
            "params": VGG_CONFIG["vgg16"],
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )

def vgg19_model_builder(num_classes: int=100, batch_norm: bool=True) -> nn.Module:
    return VGG(
        config={
            "name": "vgg19",
            "params": VGG_CONFIG["vgg19"],
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )
