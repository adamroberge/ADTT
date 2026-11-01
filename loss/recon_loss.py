from __future__ import print_function, division

from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
import numpy as np
import sys, os, pdb

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

@OPENOCC_LOSS.register_module()
class PlaneLoss(BaseLoss):
    def __init__(self, weight=1.0, reg_fff=1e-6, input_dict=None, loss_name='plane', **kwargs):
        super().__init__(weight)
        self.loss_name = str(loss_name)

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict
        self.reg_fff = reg_fff
        self.loss_func = self.plane_loss

    def plane_loss(self, logits, labels):

        xy_loss = F.mse_loss(logits[0], labels[0]) + F.l1_loss(logits[0], labels[0])
        xz_loss = F.mse_loss(logits[1], labels[1]) + F.l1_loss(logits[1], labels[1])
        yz_loss = F.mse_loss(logits[2], labels[2]) + F.l1_loss(logits[2], labels[2])


        loss = xy_loss + xz_loss + yz_loss


        return loss


@OPENOCC_LOSS.register_module()
class MaskedPlaneLoss(BaseLoss):
    def __init__(self, weight=1.0, eps=1e-6, input_dict=None, loss_name='plane_masked', **kwargs):
        super().__init__(weight)
        self.loss_name = str(loss_name)
        self.eps = float(eps)
        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels',
                'masks': 'masks',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.masked_plane_loss

    def _one_plane_loss(self, pred, target, mask):
        mask = mask.to(dtype=pred.dtype)
        if mask.dim() < pred.dim():
            mask = mask.expand_as(pred)
        valid = mask.sum()
        if float(valid.detach()) <= 0:
            return pred.new_tensor(0.0)
        l1 = (pred - target).abs() * mask
        l2 = (pred - target).pow(2) * mask
        return (l1.sum() + l2.sum()) / valid.clamp_min(self.eps)

    def masked_plane_loss(self, logits, labels, masks):
        return sum(
            self._one_plane_loss(pred_plane, gt_plane, mask_plane)
            for pred_plane, gt_plane, mask_plane in zip(logits, labels, masks)
        )


@OPENOCC_LOSS.register_module()
class PlaneWeightLoss(BaseLoss):
    def __init__(self, weight=1.0, reg_fff=1e-6, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict
        self.reg_fff = reg_fff
        self.loss_func = self.plane_loss

    def plane_loss(self, logits, labels):

        time_weights = torch.tensor([1.0, 0.85, 0.7, 0.55, 0.4, 0.25],
                                    device=logits[0].device, dtype=logits[0].dtype)
        losses = []
        T = logits[0].shape[0]
        for t in range(T):
            xy_loss = F.mse_loss(logits[0][t], labels[0][t]) + F.l1_loss(logits[0][t], labels[0][t])
            xz_loss = F.mse_loss(logits[1][t], labels[1][t]) + F.l1_loss(logits[1][t], labels[1][t])
            yz_loss = F.mse_loss(logits[2][t], labels[2][t]) + F.l1_loss(logits[2][t], labels[2][t])

            loss_t = xy_loss + xz_loss + yz_loss
            losses.append(loss_t * time_weights[t])

        loss = sum(losses) / time_weights[:T].sum()

        return loss


@OPENOCC_LOSS.register_module()
class FeatureMSELoss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred_xy': 'pred_xy',
                'pred_xz': 'pred_xz',
                'pred_yz': 'pred_yz',
                'gt_xy': 'gt_xy',
                'gt_xz': 'gt_xz',
                'gt_yz': 'gt_yz',
                'prev_steps': 'prev_steps'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.feature_mse_loss

    def feature_mse_loss(self, pred_xy, pred_xz, pred_yz, gt_xy, gt_xz, gt_yz, prev_steps):
        def compute_loss(pred, gt):
            gt_fut = gt[:, prev_steps:prev_steps + pred.shape[1]]
            return F.mse_loss(pred, gt_fut)

        loss_xy = compute_loss(pred_xy, gt_xy)
        loss_xz = compute_loss(pred_xz, gt_xz)
        loss_yz = compute_loss(pred_yz, gt_yz)

        return (loss_xy + loss_xz + loss_yz) / 3.0


@OPENOCC_LOSS.register_module()
class ReconLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100, use_weight=False, cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        self.loss_name = 'occ'

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.recon_loss
        self.ignore = ignore_label
        self.use_weight = use_weight
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None

    def recon_loss(self, logits, labels):
        weight = None
        if self.use_weight:
            if self.cls_weight is not None:
                weight = self.cls_weight
            else:
                one_hot_labels = F.one_hot(labels, num_classes=logits.shape[-1])
                cls_freq = torch.sum(one_hot_labels, dim=[0, 1, 2, 3, 4])
                weight = 1.0 / cls_freq.clamp_min_(1) * torch.numel(labels) / logits.shape[-1]

        rec_loss = F.cross_entropy(logits.permute(0, 5, 1, 2, 3, 4), labels, ignore_index=self.ignore, weight=weight)
        return rec_loss


@OPENOCC_LOSS.register_module()
class UnknownOccReconLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=-100, eps=1e-6, input_dict=None, loss_name='occ_unknown', **kwargs):
        super().__init__(weight)
        self.loss_name = str(loss_name)
        self.ignore = int(ignore_label)
        self.eps = float(eps)
        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels',
                'known_mask_xy': 'known_mask_xy',
                'known_mask_xz': 'known_mask_xz',
                'known_mask_yz': 'known_mask_yz',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.recon_loss

    def _resize_known_masks(self, known_mask_xy, known_mask_xz, known_mask_yz, occ_shape):
        target_x, target_y, target_z = [int(v) for v in occ_shape]
        batch_size, future_steps = known_mask_xy.shape[:2]

        known_xy = known_mask_xy.reshape(batch_size * future_steps, 1, known_mask_xy.shape[-2], known_mask_xy.shape[-1])
        known_xz = known_mask_xz.reshape(batch_size * future_steps, 1, known_mask_xz.shape[-2], known_mask_xz.shape[-1])
        known_yz = known_mask_yz.reshape(batch_size * future_steps, 1, known_mask_yz.shape[-2], known_mask_yz.shape[-1])

        known_xy = F.interpolate(known_xy, size=(target_x, target_y), mode='bilinear', align_corners=False)
        known_xz = F.interpolate(known_xz, size=(target_x, target_z), mode='bilinear', align_corners=False)
        known_yz = F.interpolate(known_yz, size=(target_y, target_z), mode='bilinear', align_corners=False)

        known_xy = known_xy.reshape(batch_size, future_steps, 1, target_x, target_y)
        known_xz = known_xz.reshape(batch_size, future_steps, 1, target_x, target_z)
        known_yz = known_yz.reshape(batch_size, future_steps, 1, target_y, target_z)
        return known_xy, known_xz, known_yz

    def _build_unknown_occ_mask(self, known_mask_xy, known_mask_xz, known_mask_yz, occ_shape):
        known_xy, known_xz, known_yz = self._resize_known_masks(
            known_mask_xy.to(dtype=torch.float32),
            known_mask_xz.to(dtype=torch.float32),
            known_mask_yz.to(dtype=torch.float32),
            occ_shape,
        )
        known_volume = known_xy.unsqueeze(-1) * known_xz.unsqueeze(3) * known_yz.unsqueeze(2)
        unknown_volume = 1.0 - known_volume
        return unknown_volume.clamp_(0.0, 1.0)

    def recon_loss(self, logits, labels, known_mask_xy, known_mask_xz, known_mask_yz):
        unknown_mask = self._build_unknown_occ_mask(
            known_mask_xy,
            known_mask_xz,
            known_mask_yz,
            labels.shape[-3:],
        ).to(device=logits.device)
        valid_mask = (labels != self.ignore).to(dtype=logits.dtype)
        unknown_mask = unknown_mask.to(dtype=logits.dtype) * valid_mask
        if float(unknown_mask.sum().detach()) <= 0:
            return logits.new_tensor(0.0)

        ce = F.cross_entropy(
            logits.permute(0, 5, 1, 2, 3, 4),
            labels,
            ignore_index=self.ignore,
            reduction='none',
        )
        return (ce * unknown_mask).sum() / unknown_mask.sum().clamp_min(self.eps)

@OPENOCC_LOSS.register_module()
class LovaszLoss(BaseLoss):

    def __init__(self, weight=1.0, ignore_label=None, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
        self.ignore_label = ignore_label

    def lovasz_loss(self, logits, labels):

        is_probs = False
        if logits.min() >= 0 and logits.max() <= 1 and torch.abs(logits.sum(dim=-1).mean() - 1.0) < 1e-2:
            is_probs = True

        if not is_probs:
            probs = F.softmax(logits, dim=-1)
        else:
            probs = logits

        if probs.dim() == 6:
            if probs.shape[-1] == 18:
                b, t, h, w, d, c = probs.shape
                probs = probs.reshape(b * t, h, w, d, c).permute(0, 4, 1, 2, 3).contiguous()
                labels = labels.reshape(b * t, h, w, d).contiguous()
        elif probs.dim() == 5:
            if probs.shape[-1] == 18:
                probs = probs.permute(0, 4, 1, 2, 3)
        elif probs.dim() == 4:
            if probs.shape[-1] == 18:
                probs = probs.permute(0, 3, 1, 2)

        loss = lovasz_softmax(probs, labels, ignore=self.ignore_label)
        return loss


@OPENOCC_LOSS.register_module()
class ImgLoss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'logits': 'logits',
                'labels': 'labels'
            }
        else:
            self.input_dict = input_dict

        self.loss_func = self.img_loss

        self.processor =  transforms.Compose([
            transforms.Resize((299, 297)),
            transforms.ToTensor()
        ])

    def img_loss(self, logits, labels):


        labels = [self.processor(one_label) for one_label in labels]
        labels = torch.stack(labels).to(logits.device)


        transforms.ToPILImage()(labels[0]).save("./test_results/ttttttttt/label0.png")
        transforms.ToPILImage()(logits[0]).save("./test_results/ttttttttt/logit0.png")
        transforms.ToPILImage()(labels[2]).save("./test_results/ttttttttt/label2.png")
        transforms.ToPILImage()(logits[2]).save("./test_results/ttttttttt/logit2.png")
        transforms.ToPILImage()(labels[4]).save("./test_results/ttttttttt/label4.png")
        transforms.ToPILImage()(logits[4]).save("./test_results/ttttttttt/logit4.png")

        loss = F.mse_loss(logits, labels)
        return loss


import torch
from torch.autograd import Variable
import torch.nn.functional as F
try:
    from itertools import  ifilterfalse
except ImportError:
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0 or probas.dim() < 2:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=255):
    if probas.dim() == 1:
        return probas, labels
    if probas.dim() == 2:
        pass
    elif probas.dim() == 3:
        B, N, C = probas.size()
        probas = probas.view(B * N, C)
        labels = labels.view(B * N)
    elif probas.dim() == 5:
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H * W)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(B * L * H * W, C)
        labels = labels.view(B * L * H * W)
    elif probas.dim() == 4:
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        labels = labels.view(B * H * W)
    else:
        return probas, labels
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
