from .base_loss import BaseLoss
from . import OPENOCC_LOSS
import torch
import numpy as np
import torch.nn.functional as F


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
class PoseLoss(BaseLoss):
    def __init__(
        self,
        weight=1.0,
        num_modes=3,
        input_dict=None,
        loss_type='l2',
        return_last=False,
        normalize_by_gt_mode_sum: bool = False,
        accumulate: str = 'cumsum',
        discount_gamma: float = 1.0,
        huber_beta: float = 1.0,
        warmup_steps: int = 0,
        warmup_init_mult: float = 0.0,
        clip_delta: float = 0.0,
        flow_consistency_weight: float = 0.0,
        **kwargs
    ):
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'rel_pose': 'rel_pose',
                'output_metas': 'metas',
                'hexplane': 'hexplane'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.pose_loss
        self.num_mode = num_modes
        self.loss_type = loss_type
        self.return_last = return_last
        self.normalize_by_gt_mode_sum = bool(normalize_by_gt_mode_sum)
        self.accumulate = str(accumulate)
        if self.accumulate not in ('cumsum', 'none'):
            raise ValueError(f'accumulate {self.accumulate} not supported')
        self.discount_gamma = float(discount_gamma)
        self.huber_beta = float(huber_beta)
        self.warmup_steps = int(warmup_steps)
        self.warmup_init_mult = float(warmup_init_mult)
        self.clip_delta = float(clip_delta)
        self.flow_consistency_weight = float(flow_consistency_weight)
        self.register_buffer('_step', torch.zeros((), dtype=torch.long), persistent=True)
        assert loss_type in ['l1', 'l2', 'huber'], f'loss_type {loss_type} not supported'

    def pose_loss(self, rel_pose, metas, hexplane=None):

        while rel_pose.dim() > 4 and rel_pose.shape[0] == 1:
            rel_pose = rel_pose.squeeze(0)

        if rel_pose.dim() == 3:
            rel_pose = rel_pose.unsqueeze(2)

        if rel_pose.dim() != 4:
            raise ValueError(f"Expected rel_pose to be 4D (B, F, M, 2) or 3D (B, F, 2), but got {rel_pose.dim()}D with shape {rel_pose.shape}")

        bs, num_frames, num_modes, _ = rel_pose.shape

        if self.training:
            self._step += 1

        rel_pose_T = rel_pose.transpose(1, 2)
        if self.clip_delta > 0:
            c = float(self.clip_delta)
            rel_pose_T = c * torch.tanh(rel_pose_T / c)
        if self.accumulate == 'cumsum':
            rel_pose_accum = torch.cumsum(rel_pose_T, -2)
        else:
            rel_pose_accum = rel_pose_T

        gt_rel_pose, gt_mode = [], []
        for meta in metas:
            if isinstance(meta, dict):
                 gt_rel_pose.append(meta['rel_poses'])
                 gt_mode.append(meta['gt_mode'])
            elif isinstance(meta, list):
                 poses = []
                 modes = []
                 for m in meta:
                     if isinstance(m, dict):
                         poses.append(m['rel_poses'])
                         modes.append(m['gt_mode'])
                 if poses:
                     gt_rel_pose.append(np.stack(poses))
                     gt_mode.append(np.stack(modes))

        gt_rel_pose = rel_pose.new_tensor(np.asarray(gt_rel_pose))
        gt_mode = rel_pose.new_tensor(np.asarray(gt_mode)).transpose(1,2)

        if gt_rel_pose.shape[1] > num_frames:
             gt_rel_pose = gt_rel_pose[:, -num_frames:]
             gt_mode = gt_mode[:, :, -num_frames:]

        gt_rel_pose_M = gt_rel_pose.unsqueeze(1).repeat(1, num_modes, 1, 1)
        if self.accumulate == 'cumsum':
            gt_rel_pose_accum = torch.cumsum(gt_rel_pose_M, -2)
        else:
            gt_rel_pose_accum = gt_rel_pose_M

        if self.loss_type == 'l1':
            weight = gt_mode[..., None].repeat(1, 1, 1, 2)
            traj_loss = torch.abs(rel_pose_accum - gt_rel_pose_accum) * weight
        elif self.loss_type == 'l2':
            weight = gt_mode
            traj_loss = torch.sqrt(((rel_pose_accum - gt_rel_pose_accum) ** 2).sum(-1)) * weight
        elif self.loss_type == 'huber':
            diff = rel_pose_accum - gt_rel_pose_accum
            per_xy = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=max(1e-6, float(self.huber_beta)), reduction='none')
            traj_loss = per_xy.sum(-1) * gt_mode

        final_loss = traj_loss.sum() / bs / num_frames

        if self.flow_consistency_weight > 0 and hexplane is not None:
            flow_loss = 0
            xy_feat = hexplane[0]
            B_h, F_h, C_h, H_h, W_h = xy_feat.shape

            pred_delta = rel_pose[:, :, 0, :]

            res_x, res_y = 80.0, 80.0

            for t in range(F_h - 1):
                feat_t = xy_feat[:, t]
                feat_next = xy_feat[:, t+1]

                dxy = pred_delta[:, t]

                shift_x = -dxy[:, 0] / (res_x / 2.0)
                shift_y = -dxy[:, 1] / (res_y / 2.0)

                theta = torch.eye(2, 3, device=xy_feat.device).unsqueeze(0).repeat(B_h, 1, 1)
                theta[:, 0, 2] = shift_x
                theta[:, 1, 2] = shift_y

                grid = F.affine_grid(theta, feat_t.size(), align_corners=True)
                feat_warped = F.grid_sample(feat_t, grid, align_corners=True, padding_mode='border')

                flow_loss += F.mse_loss(feat_warped, feat_next)

            final_loss += (flow_loss / max(1, F_h-1)) * self.flow_consistency_weight

        if self.warmup_steps > 0 and self.training:
            s = int(self._step.item())
            r = min(1.0, max(0.0, float(s) / float(max(1, int(self.warmup_steps)))))
            mult = float(self.warmup_init_mult) + (1.0 - float(self.warmup_init_mult)) * r
            final_loss = final_loss * mult

        return final_loss
