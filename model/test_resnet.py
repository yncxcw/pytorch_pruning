import pytest
import torch

from resnet import (
    resnet18_model_builder,
    resnet34_model_builder,
    resnet50_model_builder,
    resnet101_model_builder,
    resnet152_model_builder,
)

@pytest.mark.parametrize(
    "batch_size,batch_norm,num_classes,model_name",
    [
        (10, True, 100, "resnet18"),
        (10, False, 100, "resnet34"),
        (20, True, 50,  "resnet50"),
        (20, False, 50, "resnet101"), 
        (20, False, 50, "resnet152"),
    ]
)
def test_vgg11(batch_size, batch_norm, num_classes, model_name):
    input = torch.randn(batch_size, 3, 32, 32)
    if model_name == "resnet18":
        model = resnet18_model_builder(num_classes, True)
    elif model_name == "resnet34":
        model = resnet34_model_builder(num_classes, True)
    elif model_name == "resnet50":
        model = resnet50_model_builder(num_classes, True)
    elif model_name == "resnet101":
        model = resnet101_model_builder(num_classes, True)
    elif model_name == "resnet152":
        model = resnet152_model_builder(num_classes, True)
    else:
        raise ValueError("Unsupported model")
    output = model(input)
    assert model.name() == model_name
    assert list(output.size()) == [batch_size, num_classes]
