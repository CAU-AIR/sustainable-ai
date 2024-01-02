from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from torchvision import transforms, datasets
from .continual_dataset import ContinualDataset


__all__ = ["CASIAWebDataset", "CASIAWeb15Dataset"]

class FaceDataset(ContinualDataset):
    
    _DEFAULT_N_TASKS = None
    _N_CLASSES_PER_TASK = None
    _MEAN = (0.5)
    _STD = (0.5)
    _IMAGE_SIZE = 112

    def __init__(
        self,
        root: str,
        image_size: Optional[List] = [112, 112],
        transform: Optional[Callable] = None,
        train: Optional[bool] = True,
        normalize_targets_per_task: Optional[bool] = False,
    ) -> None:
        
        self.root = root
        self.image_size = image_size
        dataset = datasets.ImageFolder(root=root)
        self.train = train

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
        image, targets = sample

        # CASIA image resize
        if self.image_size:
            image = image.resize(self.image_size) 
        
        # to return a PIL Image
        original_img = image.copy()

        not_aug_img = self.not_aug_transform(original_img)

        image = self.transform(image)

        if self.normalize_targets_per_task and self.n_classes_per_task:
            targets -= self._current_task * self.n_classes_per_task

        return image, targets, self._current_task, not_aug_img


# @register_dataset("casiaweb")
class CASIAWebDataset(FaceDataset):
    _DATA_TYPE = "casiaweb"
    _DEFAULT_N_TASKS = 5
    _N_CLASSES_PER_TASK = 1000


class CASIAWeb15Dataset(FaceDataset):
    _DATA_TYPE = "casiaweb-15"
    _DEFAULT_N_TASKS = 5
    _N_CLASSES_PER_TASK = 3