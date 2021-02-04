"""Entry point for importing all datasets."""

from dataset.cifar100_dataset import (
    cifar100_dataloader_builder,
)


registered_datalaoders = {
    # Register dataloader    
    "cifar100": cifar100_dataloader_builder,
    # TODO: Add Imagenet dataloader
}
