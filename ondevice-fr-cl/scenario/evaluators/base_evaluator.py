from typing import Any, Dict, List, Optional, Tuple, Union
import torch

# from ...loggers import BaseLogger, Logger
from models import ContinualModel
from scenario import ClassIncremental, VerificationScenario
from utils.magic import persistent_locals


class BaseEvaluator:
    def __init__(
        self,
        method: ContinualModel,
        eval_scenario: Union[ClassIncremental, VerificationScenario],
        # logger: Optional[BaseLogger] = None,
        name: Optional[str] = "",
    ) -> "BaseEvaluator":
        """_summary_

        Args:
            method (BaseMethod): _description_
            eval_scenario (Union[ClassIncremental, TaskIncremental]): _description_
            logger (Optional[BaseLogger], optional): _description_. Defaults to None.
            device (Optional[torch.device], optional): _description_. Defaults to None.

        Returns:
            BaseEvaluator: _description_
        """
        self.name = name
        self.device = method.device

        self.method = method.to(self.device)
        self.eval_scenario = eval_scenario

        # if logger is None:
        #     logger = Logger(n_tasks=self.eval_scenario.n_tasks)
        # self.logger = logger

    @torch.no_grad()
    def _evaluate(self, task_id: int):
        pass

    def on_eval_start(self):
        """ """
        pass

    def on_eval_end(self):
        """ """
        pass

    def fit(self, task_id: int):
        pass
