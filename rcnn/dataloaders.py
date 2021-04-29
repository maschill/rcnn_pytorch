from __future__ import print_function

import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np

from rcnn import Occlusion
from rcnn import DataContainer
from rcnn import winit

CIFAR10_TRAIN_MEAN = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
CIFAR10_TRAIN_STD = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)


class CIFAR10(DataContainer):
    def __init__(self, batch_size: int, s: int = 4, mode: str = "cutout"):
        """Container for CIFAR10 dataset that provides a train, test 
            and val dataloader 

        Args:
            batch_size (int): batch size
            s (int, optional): size argument used for occlusion transform 
                                in train and test loader. Defaults to 4.
            mode (str, optional): mode of occlusion transform in train 
                                and test loader. Defaults to "cutout".
        """
        self.batch_size = batch_size
        train_ds = datasets.CIFAR10(
            root=".",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    Occlusion(mode=mode, size=s),
                    transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=winit,
        )

        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=".",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=winit,
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=".",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        Occlusion(mode=mode, size=s),
                        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=winit,
        )

        self.loader_dict = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }

        self.sizes = {
            "train": len(self.train_loader) * batch_size,
            "val": len(self.val_loader) * batch_size,
            "test": len(self.test_loader) * batch_size,
        }

    def dl_dict(self):
        """returns the dataloaders in a dict"""
        return self.loader_dict

    def update_val_loader(self, s: int, mode: str = "cutout"):
        """updates the val loader with new occlusion transformation params

        Args:
            n (int): size argument used for occlusion transform 
                    in train and test loader.
            mode (str, optional): mode of occlusion transform in train 
                                and test loader. Defaults to "cutout".
        """
        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=".",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        Occlusion(mode=mode, size=s),
                        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    ]
                ),
            ),
            batch_size=self.batch_size // 2,
            shuffle=True,
            num_workers=4,
        )
