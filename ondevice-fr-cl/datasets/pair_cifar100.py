from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100


__all__ = ["PairCIFAR100"]

class CIFARPairDataset(Dataset):
    
    _DATA_TYPE = None
    _DEFAULT_N_TASKS = None
    _MEAN = (0.5071, 0.4867, 0.4408)
    _STD = (0.2675, 0.2565, 0.2761)
    _IMAGE_SIZE = 32

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        num_pairs: int = 5000,
        data_annot: Optional[str] = None,
    ) -> None:
        assert self._DATA_TYPE in [
            "paircifar10",
            "paircifar100",
        ], "PairDataset must be subclassed and a valid _DATA_TYPE provided"

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD)
            ])
        self.transform = transform
        
        # Load dataset by args
        if self._DATA_TYPE == "paircifar10":
            self.trainset = CIFAR10(root, train=True, download=True, transform=self.transform)
            self.testset = CIFAR10(root, train=False, download=True, transform=self.transform)
        elif self._DATA_TYPE == "paircifar100":
            self.trainset = CIFAR100(root, train=True, download=True, transform=self.transform)
            self.testset = CIFAR100(root, train=False, download=True, transform=self.transform)

        # Set attributes
        self.num_pairs = num_pairs
        self.num_classes = len(self.trainset.classes)

        # data_annot = None
        if data_annot is None:
            print("Seed is not fixed, initialize indices!")
            self.get_data()
        else:
            print("Load fixed indices.")
            self.data_annot = data_annot + self._DATA_TYPE
            self.set_data_annot(self.data_annot)


    def get_data(self):
        """Get data from Args
        These comments are about cifar10 setting.
        """
        train_data, test_data = self.trainset.data.copy(), self.testset.data.copy()
        train_targets, test_targets = \
            np.array(self.trainset.targets.copy()), np.array(self.testset.targets.copy())

        # sort by class order (0, 1, 2, ...)
        train_sort_index, test_sort_index = train_targets.argsort(), test_targets.argsort()
        train_data, test_data = train_data[train_sort_index], test_data[test_sort_index]
        train_targets, test_targets = train_targets[train_sort_index], test_targets[test_sort_index]
        # number of samples
        num_pairs_per_class = self.num_pairs // self.num_classes # 500 number of pair-wise class
        retrieval_same_class = retrieval_diff_class = num_pairs_per_class // 2 # 250 same & diff class index
        # random sampling index pool for each class
        train_perm_index = np.random.permutation(train_data.shape[0] // self.num_classes) # range 0 - 5,000
        test_perm_index = np.random.permutation(test_data.shape[0] // self.num_classes) # range 0 - 1,000
        
        # 1. query index : randomly sample 500 query examples for each class in trainset
        query_index = np.random.choice(train_perm_index, num_pairs_per_class) 
        ## query index는 trainset에서 각 클래스 당 500개 씩 구성
        query_index = np.tile(query_index, self.num_classes)
        inc_index = np.array([i * (train_data.shape[0] // self.num_classes) for i in range(self.num_classes)])
        query_index = query_index + inc_index.repeat(num_pairs_per_class) # num_pairs_per_class(500) 만큼 각각의 원소 반복

        # 2. same class retrieval index : randomly sample 250 retrieval examples in same class of testset
        retrieval_same_index = np.random.choice(test_perm_index, retrieval_same_class)
        ## same class retrieval index는 testset에서 각 클래스 당 250개 씩 구성
        retrieval_same_index = np.tile(retrieval_same_index, self.num_classes)
        inc_index = np.array([i * (test_data.shape[0] // self.num_classes) for i in range(self.num_classes)])
        retrieval_same_index = retrieval_same_index + inc_index.repeat(retrieval_same_class)
        
        # 3. diff class retrieval index : randomly sample 250 retrieval examples in different class of testset
        test_index = set(range(test_data.shape[0])) # total index set of testset
        test_per_class = test_data.shape[0] // self.num_classes # 500
        test_class_index = [list(range(i*test_per_class, (i+1)*test_per_class)) for i in range(self.num_classes)]
        ## testset에서 same class를 제외한 클래스들에서 250개 sampling
        retrieval_diff_index = np.empty_like(retrieval_same_index)
        for i in range(self.num_classes):
            diff_class_index = list(test_index - set(test_class_index[i]))
            retrieval_diff_index[i*retrieval_diff_class:(i+1)*retrieval_diff_class] = np.random.choice(diff_class_index, retrieval_diff_class)
        # reshape index array for concatenating horizontally
        same_index = retrieval_same_index.reshape(self.num_classes, -1) # shape : 10, 250
        diff_index = retrieval_diff_index.reshape(self.num_classes, -1) # shape : 10, 250
        # make binary target
        same_targets = np.ones_like(retrieval_same_index).reshape(self.num_classes, -1)
        diff_targets = np.zeros_like(retrieval_diff_index).reshape(self.num_classes, -1)
        # concatnate retrieval same + diff  index
        retrieval_index = np.hstack([same_index, diff_index]).flatten()
        retrieval_targets = np.hstack([same_targets, diff_targets]).flatten()
        
        # Set data
        self.train_data, self.test_data = train_data, test_data
        self.query_index = query_index
        # Set index & binary targets
        self.retrieval_index, self.retrieval_targets = retrieval_index, retrieval_targets
        print("Completed (query index, retrieval index) shape : ({}, {})".format(len(query_index), len(retrieval_index)))
        # query_index, retrieval_index, retrieval_targets

    def set_data_annot(self, data_annot: str):
        """Load data from fixed query, retrieval index file
        """
        train_data, test_data = self.trainset.data.copy(), self.testset.data.copy()
        train_targets, test_targets = \
            np.array(self.trainset.targets.copy()), np.array(self.testset.targets.copy())

        # sort by class order (0, 1, 2, ...)
        train_sort_index, test_sort_index = train_targets.argsort(), test_targets.argsort()
        train_data, test_data = train_data[train_sort_index], test_data[test_sort_index]
        # Set data
        self.train_data, self.test_data = train_data, test_data
        
        # read data annotation file
        self.query_index = np.load(data_annot+'/query_index.npy')
        self.retrieval_index = np.load(data_annot+'/retrieval_index.npy')
        self.retrieval_targets = np.load(data_annot+'/retrieval_targets.npy')
        print("Completed (query index, retrieval index) shape : ({}, {})".format(len(self.query_index), len(self.retrieval_index)))

    def save_data_annot(self, dir: str):
        np.save(dir+'/query_index', self.query_index)
        np.save(dir+'/retrieval_index', self.retrieval_index)
        np.save(dir+'/retrieval_targets', self.retrieval_targets)

    def __len__(self):
        return len(self.query_index)

    def __getitem__(self, idx):
        query_index = self.query_index[idx]
        retrieval_index = self.retrieval_index[idx]
        retrieval_targets = self.retrieval_targets[idx]

        query_data = self.transform(self.train_data[query_index])
        retrieval_data = self.transform(self.test_data[retrieval_index])

        image = (query_data, retrieval_data)

        return query_data, retrieval_data, retrieval_targets


# @register_dataset("paircifar100")
class PairCIFAR100(CIFARPairDataset):
    _DATA_TYPE = "paircifar100"
    _DEFAULT_N_TASKS = 10 

