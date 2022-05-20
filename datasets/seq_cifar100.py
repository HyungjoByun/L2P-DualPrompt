# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize
from transformers import  ViTForImageClassification
import torch
import torch.nn as nn

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        if hasattr(self, 'features'):
            return img, target, original_img, self.features[index]

        return img, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])

    def get_data_loaders(self, download=False):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        #train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
        #                          download=download, transform=transform)
        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=download, transform=transforms.ToTensor())
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            #test_dataset = CIFAR100(base_path() + 'CIFAR100',train=False,
            #                       download=download, transform=test_transform)
            test_dataset = CIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=download, transform=transforms.ToTensor())

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        model.classifier = torch.nn.Linear(768,SequentialCIFAR100.N_CLASSES_PER_TASK* SequentialCIFAR100.N_TASKS)
        return model

    @staticmethod
    def get_loss():
        return nn.CrossEntropyLoss()

    @staticmethod
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    #! https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar100.py
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform
