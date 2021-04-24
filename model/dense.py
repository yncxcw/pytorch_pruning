"""A series of fully connected layers. Used for performance profiling."""

from functools import reduce
import math

import torch
from torch import nn


class Dense(nn.Module):

    def __init__(self, config, batch_norm: bool=False, num_classes: int=100):
        """
        Note: This model only supports cifar100 datasets.

        Args:
                config (object): Model configuration struct.
                batch_norm (bool): Whether or not to use batch_norm after conv2d.
                nm_classse (int): number of classes for classification.
        """
        super().__init__()
        self._name = config["name"]
        self._num_layers = config["num_layers"]

        input_neuros = reduce((lambda x, y: x * y), self.input_shape())
        self._extractor = nn.Linear(input_neuros, 512)
        self._layers = _make_layers(self._num_layers)
        self._classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        output = x.view(x.size()[0], -1)
        output = self._extractor(output)
        if self._num_layers > 0:
            output = self._layers(output)
        return self._classifier(output)
    
    def name(self):
        return self._name

    def input_shape(self):
        return [3, 32, 32]

def _make_layers(num_layers: int) -> nn.Sequential:
    """
    Args:
        num_layers: Numbers of dense layers to repeat.
    """
    return nn.Sequential(*[
        nn.Linear(512, 512)
        for _ in range(num_layers)    
    ])


def dense0_model_builder(num_classes: int=100, batch_norm: bool=True) -> torch.nn.Module:
    return Dense(
        config={
            "name": "dense0",
            "num_layers": 0,
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )


def dense1_model_builder(num_classes: int=100, batch_norm: bool=True) -> torch.nn.Module:
    return Dense(
        config={
            "name": "dense1",
            "num_layers": 1,
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )

def dense10_model_builder(num_classes: int=100, batch_norm: bool=True) -> torch.nn.Module:
    return Dense(
        config={
            "name": "dense10",
            "num_layers": 10,
        },
        num_classes=num_classes,
        batch_norm=batch_norm,
    )
