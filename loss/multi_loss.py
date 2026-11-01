import torch
import torch.nn as nn
from . import OPENOCC_LOSS
writer = None

@OPENOCC_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()

        assert isinstance(loss_cfgs, list)
        self.num_losses = len(loss_cfgs)

        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(OPENOCC_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)
        self.iter_counter = 0

    def forward(self, inputs):

        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            loss = loss_func(inputs)
            tot_loss += loss
            base_name = getattr(loss_func, 'loss_name', loss_func.__class__.__name__)
            key_name = str(base_name)

            count = 1
            original_key_name = key_name
            while key_name in loss_dict:
                key_name = f"{original_key_name}_{count}"
                count += 1

            loss_val = loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
            loss_dict.update({
                key_name: \
                loss_val / (loss_func.weight if loss_func.weight != 0 else 1.0)
            })
            if writer and self.iter_counter % 10 == 0:
                writer.add_scalar(
                    f'loss/{loss_func.__class__.__name__}',
                    loss_val, self.iter_counter)

        tot_loss_val = tot_loss.detach().item() if isinstance(tot_loss, torch.Tensor) else float(tot_loss)
        if writer and self.iter_counter % 10 == 0:
            writer.add_scalar(
                'loss/total', tot_loss_val, self.iter_counter)
        self.iter_counter += 1

        return tot_loss, loss_dict
