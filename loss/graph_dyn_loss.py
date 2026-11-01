import torch
from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class GraphDynLoss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None):
        if input_dict is None:
            input_dict = {'graph_dyn_loss': 'graph_dyn_loss'}
        super().__init__(weight=weight, input_dict=input_dict)
        self.loss_func = self._loss

    def _loss(self, graph_dyn_loss):
        if graph_dyn_loss is None:
            return torch.zeros((), dtype=torch.float32)
        if isinstance(graph_dyn_loss, (float, int)):
            return torch.as_tensor(graph_dyn_loss, dtype=torch.float32)
        if not torch.is_tensor(graph_dyn_loss):
            return torch.as_tensor(graph_dyn_loss)
        return graph_dyn_loss
