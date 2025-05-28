from typing import Dict, Union

import torch
import torch.nn as nn

from .builder import LOSS


@LOSS
class Regression_Loss(nn.Module):
    def __init__(self, p: int = 1):
        super().__init__()
        if p == 1:
            self.criterion = nn.L1Loss(reduction="none")
        elif p == 2:
            self.criterion = nn.MSELoss()
        else:
            raise ValueError
        
    def forward(self, predictions: Dict[str, torch.Tensor], ground_truth: Dict[str, Union[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        ys_gt = ground_truth["ys"]
        us = ground_truth["us"]
        vs = ground_truth["vs"]

        ys_pred = predictions["e2e"][us, vs]

        loss = self.criterion(ys_pred, ys_gt)

        if "weight" in ground_truth:
            loss = loss * ground_truth["weight"]
        
        loss = loss.mean()

        return {"loss": loss}