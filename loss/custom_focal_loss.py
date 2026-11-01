import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_loss import BaseLoss
from . import OPENOCC_LOSS
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight

    loss = loss.sum(-1).mean()
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.sum(-1).mean()
    return loss


@OPENOCC_LOSS.register_module()
class CustomFocalLoss(BaseLoss):

    def __init__(self,
                 weight=1.0,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False,
                 bev_h=200,
                 bev_w=200,
                 input_dict=None,
                 **kwargs
                 ):
        super(CustomFocalLoss, self).__init__(weight=weight, input_dict=input_dict)
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        self.bev_h = bev_h
        self.bev_w = bev_w

        xy, yx = torch.meshgrid([torch.arange(bev_h) - bev_h / 2, torch.arange(bev_w) - bev_w / 2])
        c = torch.stack([xy, yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c / c_max + 1)

        self.loss_func = self.compute_loss

    def compute_loss(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                ignore_index=255,
                reduction_override=None):
        if self.c.device != pred.device:
            self.c = self.c.to(pred.device)


        if pred.dim() == 5 and target.dim() == 5 and pred.shape[0] != target.shape[0]:
             target = target.reshape(-1, *target.shape[2:])

        B = target.shape[0]
        H, W, D = target.shape[1], target.shape[2], target.shape[3]


        c = self.c[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)

        visible_mask = (target != ignore_index).reshape(-1).nonzero().squeeze(-1)
        if weight is not None:
            weight_mask = weight[None, :] * c[visible_mask, None]
        else:
            weight_mask = c[visible_mask, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)
        pred = pred[visible_mask]

        target = target.reshape(-1)[visible_mask]

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target.to(torch.long),
                weight_mask,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
