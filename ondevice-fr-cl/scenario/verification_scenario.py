# This code was written by Wonseon Lim.

from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import get_metrics


class VerificationScenario:
    def __init__(
        self,
        dataset: Dataset,
        n_tasks: int,
        batch_size: int,
        n_workers: Optional[int] = 0,
    ) -> "VerificationScenario":
        """_summary_

        Args:
            dataset (Dataset): _description_
            n_tasks (int): _description_
            batch_size (int): _description_
            n_workers (Optional[int], optional): _description_. Defaults to 0.

        Returns:
            VerificationScenario: _description_
        """        

        self.dataset = dataset
        self.n_tasks = n_tasks # n_tasks is not used in this class.
        self.batch_size = batch_size
        self.n_workers = n_workers

        self._task_id = 0
        self.loader = self._create_dataloader(dataset=self.dataset)

    @property
    def n_samples(self) -> int:
        """Total number of samples in the whole continual setting."""
        return len(self.dataset)

    @property
    def n_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        if isinstance(self.dataset, torch.utils.data.Subset):
            targets = np.array(self.dataset.testset.targets)[self.dataset.indices]
        else:
            targets = self.dataset.testset.targets

        return len(np.unique(targets))

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False
        )
        return loader

    def evaluate(self, feats, 
            FPRs=['1e-4', '5e-4', '1e-3', '5e-3', '5e-2']):
        # CIFAR: ResNet18[batch, 2, 256], Face: IResNet50[batch, 2, 512]
        dim = feats.shape[-1]

        # pair-wise scores
        feats = F.normalize(feats.reshape(-1, dim), dim=1)
        feats = feats.reshape(-1, 2, dim)
        feats0 = feats[:, 0, :]
        feats1 = feats[:, 1, :]
        scores = torch.sum(feats0 * feats1, dim=1).tolist()

        return get_metrics(self.dataset.retrieval_targets, scores, FPRs)