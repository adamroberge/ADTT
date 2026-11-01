from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
import torch


@OPENOCC_LOSS.register_module()
class TemporalConsistencyLoss(BaseLoss):

    def __init__(self, weight=1.0, mode='l1', input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred_output': 'pred_output',
                'gt_output': 'gt_output'
            }
        else:
            self.input_dict = input_dict

        self.mode = mode
        self.loss_func = self.temporal_consistency_loss

    def temporal_consistency_loss(self, pred_output, gt_output):
        if len(pred_output.shape) != 5:
            return torch.tensor(0.0, device=pred_output.device, requires_grad=True)

        T = pred_output.shape[0]
        if T < 2:
            return torch.tensor(0.0, device=pred_output.device, requires_grad=True)

        pred_probs = F.softmax(pred_output, dim=-1)
        pred_classes = pred_output.argmax(dim=-1)

        if len(gt_output.shape) == 4:
            gt_classes = gt_output.long()
        elif gt_output.shape[-1] > 1:
            gt_classes = gt_output.argmax(dim=-1)
        else:
            gt_classes = gt_output.squeeze(-1).long()

        pred_diff = (pred_classes[1:] != pred_classes[:-1]).float()
        gt_diff = (gt_classes[1:] != gt_classes[:-1]).float()

        non_empty_mask_prev = (gt_classes[:-1] != 17)
        non_empty_mask_curr = (gt_classes[1:] != 17)
        valid_mask = non_empty_mask_prev & non_empty_mask_curr

        change_mask = valid_mask & ((gt_diff > 0) | (pred_diff > 0))

        if change_mask.sum() > 0:
            if self.mode == 'l1':
                loss = F.l1_loss(pred_diff[change_mask], gt_diff[change_mask])
            else:
                loss = F.mse_loss(pred_diff[change_mask], gt_diff[change_mask])
        else:
            loss = torch.tensor(0.0, device=pred_output.device, requires_grad=True)

        return loss


@OPENOCC_LOSS.register_module()
class TriplaneFeatureTemporalSmoothLoss(BaseLoss):
    def __init__(self, weight=1.0, mode='l1', input_dict=None, **kwargs):
        super().__init__(weight)
        if input_dict is None:
            self.input_dict = {
                'feat_maps': 'feat_maps',
            }
        else:
            self.input_dict = input_dict
        self.mode = str(mode)
        self.loss_func = self.triplane_smooth_loss

    def triplane_smooth_loss(self, feat_maps):
        device = torch.device('cpu')
        if isinstance(feat_maps, (list, tuple)) and len(feat_maps) > 0 and torch.is_tensor(feat_maps[0]):
            device = feat_maps[0].device
        if feat_maps is None:
            return torch.zeros((), device=device, requires_grad=True)
        if not isinstance(feat_maps, (list, tuple)) or len(feat_maps) != 3:
            return torch.zeros((), device=device, requires_grad=True)
        loss = 0.0
        count = 0
        for fm in feat_maps:
            if not torch.is_tensor(fm) or fm.dim() != 4:
                continue
            if fm.shape[0] < 2:
                continue
            diff = fm[1:] - fm[:-1]
            if self.mode == 'l2':
                loss = loss + torch.mean(diff * diff)
            else:
                loss = loss + torch.mean(torch.abs(diff))
            count += 1
        if count == 0:
            return torch.zeros((), device=device, requires_grad=True)
        return loss / float(count)
