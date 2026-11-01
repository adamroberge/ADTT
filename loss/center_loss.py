import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss
from mmengine.registry import MODELS

@MODELS.register_module()
class CenterLoss(BaseLoss):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight

    def forward(self, inputs):
        loss = inputs.get('center_loss', None)
        if loss is None:
            return torch.tensor(0.0, device=inputs['feat_maps'][0].device)

        return loss * self.weight
