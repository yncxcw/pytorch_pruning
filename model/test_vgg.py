import pytest
import torch

from vgg import (
    VGG,
    VGG_CONFIG
)

@pytest.mark.parametrize(
    "batch_size,batch_norm,num_classes,model_name",
    [
        (10, True, 100, "vgg11"),
        (10, False, 100, "vgg13"),
        (20, True, 50, "vgg16"),
        (20, False, 50, "vgg19"),
    ]
)
def test_vgg11(batch_size, batch_norm, num_classes, model_name):
    input = torch.randn(batch_size, 3, 32, 32)
    model = VGG(
        config={
            "name": model_name,
            "params": VGG_CONFIG[model_name],
        },
        batch_norm=batch_norm,
        num_classes=num_classes
    )
    output = model(input)
    assert model.name() == model_name
    assert list(output.size()) == [batch_size, num_classes]
