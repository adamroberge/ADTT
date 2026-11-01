from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import torch


@OPENOCC_LOSS.register_module()
class SceLoss(BaseLoss):
    def __init__(self, weight=1.0, ignore_label=-100,
            use_weight=False, cls_weight=None, input_dict=None,
            temperature=0.07, alpha=3, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'ce_inputs': 'ce_inputs',
                'ce_labels': 'ce_labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.sce_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
        self.temperature = temperature
        self.alpha = alpha


    def sce_loss(self, x, y, alpha=None):
        if alpha is None:
            alpha = self.alpha


        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        weight = 1-F.normalize(cos(x, y),p=2,dim=-1)

        cosine_sim = (x * y).sum(dim=-1)

        cosine_sim = cosine_sim / self.temperature

        loss = (1 - cosine_sim).pow_(alpha)


        loss = loss.mean()
        return loss
