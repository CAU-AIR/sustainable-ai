import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .base_evaluator import BaseEvaluator
# from ...loggers import BaseLogger, Logger
from models import ContinualModel
from scenario import VerificationScenario

class VerificationEvaluator(BaseEvaluator):
    def __init__(
        self,
        method: ContinualModel,
        eval_scenario: VerificationScenario,
        # logger: Optional[BaseLogger] = None,
        name: Optional[str] = "Verification"
    ) -> "VerificationEvaluator":
        """_summary_

        Args:
            method (ContinualModel): _description_
            eval_scenario (VerificationScenario): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            name (Optional[str], optional): _description_. Defaults to "Rep".

        Returns:
            VerificationEvaluator: _description_
        """        
        super().__init__(method, eval_scenario, name) # logger

        # To log metrics by face dataset name(lfw, calfw, cplfw, agedb_30)
        if self.eval_scenario.dataset._DATA_TYPE in ["lfw", "calfw", "cplfw", "agedb_30"]:
            self.name = self.name + self.eval_scenario.dataset._DATA_TYPE

    @torch.no_grad()
    def _evaluate(self, task_id: int) -> List[float]:
        """_summary_

        Args:
            task_id (int): _description_

        Returns:
            _type_: _description_
        """
        
        mb_size = self.eval_scenario.batch_size
        # need feats according to network output dimension
        feats = torch.zeros([self.eval_scenario.n_samples, 2, self.method.net.output_dim], dtype=torch.float32).to(self.device)

        for idx, (query_x, retrieval_x, y) in enumerate(self.eval_scenario.loader):
            query_x, retrieval_x, y = query_x.to(self.device), retrieval_x.to(self.device), y.to(self.device)

            qeury_feat = self.method.net(query_x, outputs='features')
            # query_x = torch.flip(query_x, [3]) # flip
            # qeury_feat += self.method.net.backbone(query_x)
            retrieval_feat = self.method.net(retrieval_x, outputs='features')
            # retrieval_x = torch.flip(retrieval_x, [3]) # flip
            # retrieval_feat += self.method.net.backbone(retrieval_x)

            feats[(idx*mb_size):(idx+1)*mb_size, 0, :] = qeury_feat
            feats[(idx*mb_size):(idx+1)*mb_size, 1, :] = retrieval_feat

        results = self.eval_scenario.evaluate(feats.cpu())
        results = dict(results)
        metric = ['ACC', 'AUC']
        
        return results[metric[0]], results[metric[1]]

    def on_eval_start(self):
        """ """
        self.method.net.eval()

    def on_eval_end(self, tasks_acc: float, tasks_auc: float, current_task_id: int) -> None:
        """Representation Evluation Setting is not divided into tasks.
           Just one task of pair-wise classification.
        """
        msg = "\n" + f"Verification | {self.name}-ACC: {tasks_acc:.2f} | {self.name}-AUC: {tasks_auc:.2f} |"
        # self.logger.write_txt(msg=msg)
        print(msg)
        
        self.method.net.train()

    def fit(self, current_task_id: int, logger: str) -> None:
        """_summary_

        Args:
            current_task_id (int): _description_
        """
        # self.logger = logger

        self.on_eval_start()

        tasks_acc, tasks_auc = self._evaluate(task_id=current_task_id)

        self.on_eval_end(tasks_acc=tasks_acc, tasks_auc=tasks_auc, current_task_id=current_task_id)

        return tasks_acc