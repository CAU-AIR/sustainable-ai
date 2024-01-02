from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


__all__ = ["LFWPairDataset", "CALFWPairDataset", "CPLFWPairDataset", "AGEDB30PairDataset"]

class PairFaceDataset(Dataset):
    _DATA_TYPE = None
    _DEFAULT_N_TASKS = None
    _MEAN = (0.5)
    _STD = (0.5)
    _IMAGE_SIZE = 112
    
    def __init__(
        self,
        root: str,
        image_size: Optional[List] = [112, 112],
        metrics: Optional[List] = ['ACC'],
        transform: Optional[Callable] = None,
        data_annot: Optional[str] = None,
    ) -> None:
        assert self._DATA_TYPE in [
            "lfw",
            "calfw",
            "cplfw",
            "agedb_30",
        ], "PairFaceDataset must be subclassed and a valid _DATA_TYPE provided"
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD)
            ])
        self.target_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.transform = transform
        
        self.root = root # '../cl-dataset/
        self.image_size = image_size
        self.metrics = metrics
        self.data_annot = data_annot # '../cl-dataset/{}_ann.txt

        print(f"Load {self._DATA_TYPE} annotation file.")
        self.data_annot = data_annot + self._DATA_TYPE + "_ann.txt"

        self.data, self.targets = self.get_dataset()
        self.retrieval_targets = self.targets

    def get_dataset(self):
        with open(self.data_annot, 'r') as f:
            lines = f.readlines()

        data, targets = [], []
        for line in lines:
            target, img, retrieval_img = line.rstrip().split(' ')
            img, retrieval_img = self.root + img, self.root + retrieval_img
            data.append((img, retrieval_img))
            targets.append(int(target))
        return data, np.array(targets, dtype=np.int8) 

    def _loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is `0` for different indentities and `1` for same identities.
        """
        img1, img2 = self.data[index]
        img1, img2 = self._loader(img1), self._loader(img2)
        target = self.targets[index]

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        return img1, img2, target #.long()


# @register_dataset("lfw")
class LFWPairDataset(PairFaceDataset):
    _DATA_TYPE = "lfw"
    _DEFAULT_N_TASKS = 5 

# @register_dataset("calfw")
class CALFWPairDataset(PairFaceDataset):
    _DATA_TYPE = "calfw"
    _DEFAULT_N_TASKS = 5 

# @register_dataset("cplfw")
class CPLFWPairDataset(PairFaceDataset):
    _DATA_TYPE = "cplfw"
    _DEFAULT_N_TASKS = 5 

# @register_dataset("agedb_30")
class AGEDB30PairDataset(PairFaceDataset):
    _DATA_TYPE = "agedb_30"
    _DEFAULT_N_TASKS = 5 

