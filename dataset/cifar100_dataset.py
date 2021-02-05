"""Training dataset and validation dataset."""

import os
from typing import Any, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

 
class Cifar100Dataset(Dataset):
    """Pytorch dataset wrapping around CIFAR100."""

    def __init__(self, path: str, type: str="train", transform=None):
        if path is None:
            raise ValueError("Path to the dataset can't be empty.")

        self._images = None
        self._labels = []
        # Chekcing https://www.cs.toronto.edu/~kriz/cifar.html for details
        with open(os.path.join(path, type), "rb") as f:
            # dict contains `data` and `labels`
            self._dataset = pickle.load(f, encoding="latin1")
            print(self._dataset.keys())
            if "labels" in self._dataset:
                self._labels.extend(self._dataset["labels"])
            else:
                self._labels.extend(self._dataset["fine_labels"])
            
            # Image with shape NCHW
            self._images = self._dataset["data"].reshape(-1, 3, 32, 32)
            # Transpose to NCHW
            self._images = self._images.transpose((0, 2, 3, 1))
        
        meta_file = os.path.join(path, "meta")
        # Load meta file
        with open(meta_file, "rb") as f:
            meta = pickle.load(f, encoding="latin1")
            classes = meta["fine_label_names"]
            self._class_to_idx = {_class: i for i, _class in enumerate(classes)}

        self._transform = transform

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        image, label = self._images[index], self._labels[index]
        
        image = Image.fromarray(image)
        if self._transform is not None:
            image = self._transform(image)

        return image, label


class Cifar100TrainDataset(Cifar100Dataset):
    """Training dataset implementation"""

    def __init__(self, path: str, transform):
        super().__init__(path, "train", transform)


class Cifar100TestDataset(Cifar100Dataset):
    """Test dataset implementation"""

    def __init__(self, path: str, transform):
        super().__init__(path, "test", transform)


def cifar100_dataloader_builder(path:str, type:str, batch_size:int, shuffle:bool=True, num_workers:int=5) -> torch.utils.data.DataLoader:
    """Function to build pytorch dataloader for Cifar100 dataset."""

    # See https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    if type == "train":
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # ToTensor alsos normalize the image to [0, 1.0).
            transforms.ToTensor(),
            # transforms.Normalize(
            #     CIFAR100_TRAIN_MEAN,
            #     CIFAR100_TRAIN_STD,
            # )
        ])
        dataset = Cifar100TrainDataset(
            path=path,
            transform=transform,
        )
    elif type == "test":
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     CIFAR100_TRAIN_MEAN,
            #     CIFAR100_TRAIN_STD,
            # )
        ])
        dataset = Cifar100TestDataset(
            path=path,
            transform=transform,
        )
    else:
        raise ValueError("Not supported dataset type {}".format(type))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    return dataloader
    
    
