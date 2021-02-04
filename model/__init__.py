from model.vgg import (
    vgg11_model_builder,
    vgg13_model_builder,
    vgg16_model_builder,
    vgg19_model_builder,
)

registered_models = {
    "vgg11": vgg11_model_builder,
    "vgg13": vgg13_model_builder,
    "vgg16": vgg16_model_builder,
    "vgg19": vgg19_model_builder,
}

