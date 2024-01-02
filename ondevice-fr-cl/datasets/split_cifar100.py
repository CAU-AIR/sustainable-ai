from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from .continual_dataset import ContinualDataset


__all__ = ["SplitCIFAR100"]

class CIFARDataset(ContinualDataset):

    _DATA_TYPE = None
    _DEFAULT_N_TASKS = None
    _N_CLASSES_PER_TASK = None
    _MEAN = (0.5071, 0.4867, 0.4408)
    _STD = (0.2675, 0.2565, 0.2761)
    _IMAGE_SIZE = 32
                                  
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        normalize_targets_per_task: Optional[bool] = False,
        train: Optional[bool] = True,
        download: Optional[bool] = True,
    ) -> None:
        
        assert self._DATA_TYPE in [
            "cifar10",
            "cifar100",
        ], "CIFARDataset must be subclassed and a valid _DATA_TYPE provided"

        if self._DATA_TYPE == "cifar10":
            dataset = CIFAR10(root, train=train, download=download)
        if self._DATA_TYPE == "cifar100":
            dataset = CIFAR100(root, train=train, download=download)

        super().__init__(
            dataset=dataset,
            transform=transform,
            normalize_targets_per_task=normalize_targets_per_task,
        )
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
   
    def __getitem__(self, index: int):
        assert index >= 0 and index < len(
            self.dataset
        ), f"Provided index ({index}) is outside of dataset range."

        sample = self.dataset[index]
        data, targets = sample

        # to return a PIL Image
        original_img = data.copy()

        not_aug_img = self.not_aug_transform(original_img)

        data = self.transform(data)

        if self.normalize_targets_per_task and self.n_classes_per_task:
            targets -= self._current_task * self.n_classes_per_task

        return data, targets, self._current_task, not_aug_img


# @register_dataset("cifar100")
class SplitCIFAR100(CIFARDataset):
    _DATA_TYPE = "cifar100"
    _DEFAULT_N_TASKS = 10  
    _N_CLASSES_PER_TASK = 10     
    