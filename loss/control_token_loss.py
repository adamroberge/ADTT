from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch
import torch.nn.functional as F


@OPENOCC_LOSS.register_module()
class LearnableControlTokenLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        diversity_weight=1.0,
        separation_weight=0.5,
        input_dict=None,
        **kwargs,
    ):
        super().__init__(weight)
        self.loss_name = 'ctrl'
        if input_dict is None:
            self.input_dict = {
                'learnable_tokens': 'learnable_control_tokens',
                'real_tokens': 'real_control_tokens',
            }
        else:
            self.input_dict = input_dict
        self.diversity_weight = float(diversity_weight)
        self.separation_weight = float(separation_weight)
        self.loss_func = self._loss

    def _loss(self, learnable_tokens, real_tokens):
        q = int(learnable_tokens.shape[2])
        learnable = F.normalize(learnable_tokens, dim=-1)
        real = F.normalize(real_tokens, dim=-1)

        sim = torch.matmul(learnable, learnable.transpose(-1, -2))
        eye = torch.eye(q, dtype=sim.dtype, device=sim.device).view(1, 1, q, q)
        diversity = ((sim - eye) ** 2 * (1.0 - eye)).sum() / max(1, q * max(1, q - 1) * learnable.shape[0] * learnable.shape[1])

        cross = torch.matmul(learnable, real.transpose(-1, -2))
        separation = (cross ** 2).mean()

        return self.diversity_weight * diversity + self.separation_weight * separation
