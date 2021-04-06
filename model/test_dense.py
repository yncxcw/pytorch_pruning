import pytest
import torch

from dense import Dense

@pytest.mark.parametrize(
    "num_layers",
    [2, 3, 10],
)
def test_dense(num_layers):
    input = torch.rand(10, 3, 32, 32)
    model_name = "dense-" + str(num_layers)
    model = Dense(
        config={
            "name": model_name,
            "num_layers": num_layers,
        },
        num_classes=100,
    )
    output = model(input)
    assert model.name() == model_name
    assert list(output.size()) == [10, 100]
