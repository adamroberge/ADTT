import torch
import torch.nn as nn
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn.functional as F
from copy import deepcopy


import sys, os, pdb
try:
    from utils.dynamic_sparse_graph_optimized import generate_adaptive_graph_optimized
except Exception:
    generate_adaptive_graph_optimized = None

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def compose_triplane_channelwise(feat_maps):
    h_xy, h_xz, h_yz = feat_maps
    assert h_xy.shape[1] == h_xz.shape[1] == h_yz.shape[1]
    C, H, W = h_xy.shape[-3:]
    D = h_xz.shape[-1]

    newH = max(H, W)
    newW = max(W, D)
    h_xy = F.pad(h_xy, (0, newW - W, 0, newH - H))
    h_xz = F.pad(h_xz, (0, newW - D, 0, newH - H))
    h_yz = F.pad(h_yz, (0, newW - D, 0, newH - W))
    h = torch.cat([h_xy, h_xz, h_yz], dim=1)
    return h, (H, W, D)

def decompose_triplane_channelwise(composed_map, sizes):
    H, W, D = sizes
    C = composed_map.shape[1] // 3
    h_xy = composed_map[:, :C, :H, :W]
    h_xz = composed_map[:, C:2*C, :H, :D]
    h_yz = composed_map[:, 2*C:, :W, :D]
    return h_xy, h_xz, h_yz


@MODELS.register_module()
class TripLane(BaseModule):
    def __init__(
            self,
            encoder_cfg,
            decoder_cfg,
            num_classes=18,
            expansion=8,
            only_triplane=False,
            pool_type: str = 'mean',
            plane_act: str = 'tanh',
            plane_act_scale: float = 0.5,
            use_static_dynamic_split: bool = True,
            static_channels: int = None,
            dynamic_channels: int = None,
            channel_partition: str = 'fixed',
            dynamic_norm: str = 'none',
            dynamic_act: str = 'none',
            dynamic_act_scale: float = 1.0,
            temporal_fusion: str = 'none',
            temporal_kernel: int = 3,
            temporal_layers: int = 1,
            use_graph_dyn: bool = False,
            graph_dyn_cfg='auto',
            graph_dyn_planes=('xy', 'xz', 'yz'),
            graph_dyn_loss_planes=('xy', 'xz', 'yz'),
            graph_dyn_high_motion_top_ratio: float = 0.20,
            graph_dyn_spatial_radius: int = 2,
            graph_dyn_temporal_radius: int = 1,
            graph_dyn_k_additional: int = 16,
            graph_dyn_use_ego_motion_compensation: bool = True,
            graph_dyn_ego_dropout: float = 0.0,
            graph_dyn_fuse: str = 'gated',
            graph_dyn_scale: float = 1.0,
            graph_dyn_inference_use_dynamic_edges: bool = False,
            vqvae_cfg=None,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes
        self.pool_type = str(pool_type)
        self.plane_act = str(plane_act)
        self.plane_act_scale = float(plane_act_scale)
        self.dynamic_norm_type = str(dynamic_norm)
        self.dynamic_act = str(dynamic_act)
        self.dynamic_act_scale = float(dynamic_act_scale)
        self.temporal_fusion = str(temporal_fusion)
        self.temporal_kernel = int(temporal_kernel)
        self.temporal_layers = int(temporal_layers)

        self.use_static_dynamic_split = bool(use_static_dynamic_split)
        if self.use_static_dynamic_split:
            if static_channels is None and dynamic_channels is None:
                dc = int(self.expansion) // 2
                sc = int(self.expansion) - dc
            else:
                sc = int(static_channels) if static_channels is not None else int(self.expansion) - int(dynamic_channels)
                dc = int(dynamic_channels) if dynamic_channels is not None else int(self.expansion) - int(static_channels)
            if sc <= 0 or dc <= 0 or (sc + dc) != int(self.expansion):
                raise ValueError(f'invalid static/dynamic channels: static={sc}, dynamic={dc}, expansion={self.expansion}')
        else:
            sc = int(self.expansion)
            dc = 0
        self.static_channels = int(sc)
        self.dynamic_channels = int(dc)

        self.channel_partition = str(channel_partition)
        if self.channel_partition == 'fixed':
            self.partition_proj = nn.Identity()
        elif self.channel_partition == 'mix':
            self.partition_proj = nn.Conv2d(int(self.expansion), int(self.expansion), kernel_size=1, bias=False)
            with torch.no_grad():
                eye = torch.eye(int(self.expansion), dtype=self.partition_proj.weight.dtype, device=self.partition_proj.weight.device)
                self.partition_proj.weight.copy_(eye.view(int(self.expansion), int(self.expansion), 1, 1))
        else:
            raise ValueError(f'unsupported channel_partition: {self.channel_partition}')

        self.norm_static = nn.InstanceNorm2d(self.static_channels)
        if self.dynamic_channels > 0:
            if self.dynamic_norm_type == 'none':
                self.norm_dynamic = nn.Identity()
            elif self.dynamic_norm_type == 'group1':
                self.norm_dynamic = nn.GroupNorm(1, self.dynamic_channels, eps=1e-6, affine=True)
            elif self.dynamic_norm_type == 'instance':
                self.norm_dynamic = nn.InstanceNorm2d(self.dynamic_channels)
            else:
                raise ValueError(f'unsupported dynamic_norm: {self.dynamic_norm_type}')
            self.dynamic_scale = nn.Parameter(torch.ones(1, self.dynamic_channels, 1, 1))
        else:
            self.norm_dynamic = nn.Identity()
            self.dynamic_scale = None

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)

        self.geo_convs = TriplaneGroupResnetBlock(
            expansion, out_channels=64,
            ks=5, input_norm=False, input_act=False)

        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.pos_num_freq = 6

        self.only_triplane = only_triplane

        self.use_graph_dyn = bool(use_graph_dyn) and self.use_static_dynamic_split and (self.dynamic_channels > 0) and (not bool(self.only_triplane))
        if self.use_graph_dyn:
            if graph_dyn_cfg == 'auto':
                self.graph_dyn_cfg = dict(
                    type='GraphMAE',
                    in_dim=int(self.dynamic_channels),
                    num_hidden=32,
                    num_layers=2,
                    nhead=4,
                    nhead_out=1,
                    activation='prelu',
                    feat_drop=0.2,
                    attn_drop=0.1,
                    negative_slope=0.2,
                    residual=False,
                    encoder_type='gat',
                    decoder_type='mlp',
                    mask_rate=0.5,
                    norm=None,
                    loss_fn='sce',
                    drop_edge_rate=0.0,
                    replace_rate=0.1,
                    alpha_l=2,
                    concat_hidden=False,
                    num_frames=1,
                    use_change_based_masking=True,
                    use_position_encoding=True,
                    pos_encoding_type='learnable',
                    pos_encoding_dim=3,
                    pos_encoding_scale=1.0,
                    use_dynamic_head=True,
                    dynamic_head_hidden_dim=16,
                    dynamic_head_weight=0.1,
                    dynamic_high_quantile=0.7,
                    dynamic_low_quantile=0.5,
                )
            else:
                self.graph_dyn_cfg = graph_dyn_cfg
            self.graph_dyn_planes = tuple(graph_dyn_planes) if graph_dyn_planes is not None else tuple()
            self.graph_dyn_loss_planes = tuple(graph_dyn_loss_planes) if graph_dyn_loss_planes is not None else tuple()
            self.graph_dyn_high_motion_top_ratio = float(graph_dyn_high_motion_top_ratio)
            self.graph_dyn_spatial_radius = int(graph_dyn_spatial_radius)
            self.graph_dyn_temporal_radius = int(graph_dyn_temporal_radius)
            self.graph_dyn_k_additional = int(graph_dyn_k_additional)
            self.graph_dyn_use_ego_motion_compensation = bool(graph_dyn_use_ego_motion_compensation)
            self.graph_dyn_ego_dropout = float(graph_dyn_ego_dropout)
            self.graph_dyn_fuse = str(graph_dyn_fuse)
            if self.graph_dyn_fuse not in ('gated', 'add', 'replace'):
                raise ValueError(f'unsupported graph_dyn_fuse: {self.graph_dyn_fuse}')
            self.graph_dyn_scale = float(graph_dyn_scale)
            self.graph_dyn_inference_use_dynamic_edges = bool(graph_dyn_inference_use_dynamic_edges)

            if self.graph_dyn_cfg is not None:
                if generate_adaptive_graph_optimized is None:
                    raise ImportError('utils.dynamic_sparse_graph_optimized.generate_adaptive_graph_optimized is required for graph_dyn_cfg')
                self.graph_dyn = MODELS.build(self.graph_dyn_cfg)
                gdim = int(getattr(self.graph_dyn, 'output_hidden_dim', self.dynamic_channels))
                if gdim == int(self.dynamic_channels):
                    self.graph_dyn_proj = nn.Identity()
                else:
                    self.graph_dyn_proj = nn.Linear(gdim, int(self.dynamic_channels), bias=False)
                if self.graph_dyn_fuse == 'gated':
                    self.graph_dyn_gate_xy = nn.Parameter(torch.tensor(-2.0))
                    self.graph_dyn_gate_xz = nn.Parameter(torch.tensor(-2.0))
                    self.graph_dyn_gate_yz = nn.Parameter(torch.tensor(-2.0))
                else:
                    self.graph_dyn_gate_xy = None
                    self.graph_dyn_gate_xz = None
                    self.graph_dyn_gate_yz = None
            else:
                self.graph_dyn = None
                self.graph_dyn_proj = None
                self.graph_dyn_gate_xy = None
                self.graph_dyn_gate_xz = None
                self.graph_dyn_gate_yz = None
        else:
            self.graph_dyn_cfg = None
            self.graph_dyn_planes = tuple()
            self.graph_dyn_loss_planes = tuple()
            self.graph_dyn_high_motion_top_ratio = float(graph_dyn_high_motion_top_ratio)
            self.graph_dyn_spatial_radius = int(graph_dyn_spatial_radius)
            self.graph_dyn_temporal_radius = int(graph_dyn_temporal_radius)
            self.graph_dyn_k_additional = int(graph_dyn_k_additional)
            self.graph_dyn_use_ego_motion_compensation = bool(graph_dyn_use_ego_motion_compensation)
            self.graph_dyn_ego_dropout = float(graph_dyn_ego_dropout)
            self.graph_dyn_fuse = str(graph_dyn_fuse)
            if self.graph_dyn_fuse not in ('gated', 'add', 'replace'):
                raise ValueError(f'unsupported graph_dyn_fuse: {self.graph_dyn_fuse}')
            self.graph_dyn_scale = float(graph_dyn_scale)
            self.graph_dyn_inference_use_dynamic_edges = bool(graph_dyn_inference_use_dynamic_edges)
            self.graph_dyn = None
            self.graph_dyn_proj = None
            self.graph_dyn_gate_xy = None
            self.graph_dyn_gate_xz = None
            self.graph_dyn_gate_yz = None

        if self.pool_type == 'attn':
            self.pool_logits = nn.Conv3d(expansion, 1, kernel_size=1, stride=1, padding=0, bias=True)
        elif self.pool_type == 'mean':
            self.pool_logits = None
        else:
            raise ValueError(f'unsupported pool_type: {self.pool_type}')

        if self.temporal_fusion == 'dwconv':
            k = self.temporal_kernel
            if k <= 0 or k % 2 == 0:
                raise ValueError(f'temporal_kernel must be positive odd, got {k}')
            if self.temporal_layers <= 0:
                raise ValueError(f'temporal_layers must be > 0, got {self.temporal_layers}')
            layers = []
            for _ in range(self.temporal_layers):
                layers.append(nn.Conv1d(expansion, expansion, kernel_size=k, padding=k // 2, groups=expansion, bias=True))
            self.temporal_dwconv = nn.Sequential(*layers)
        elif self.temporal_fusion == 'none':
            self.temporal_dwconv = None
        else:
            raise ValueError(f'unsupported temporal_fusion: {self.temporal_fusion}')

    def _graph_dyn_gate(self, plane_key: str) -> torch.Tensor:
        if self.graph_dyn_fuse != 'gated':
            return torch.ones((), device=self.dynamic_scale.device, dtype=self.dynamic_scale.dtype)
        if plane_key == 'xy':
            return torch.sigmoid(self.graph_dyn_gate_xy)
        if plane_key == 'xz':
            return torch.sigmoid(self.graph_dyn_gate_xz)
        if plane_key == 'yz':
            return torch.sigmoid(self.graph_dyn_gate_yz)
        raise ValueError(f'unsupported plane_key: {plane_key}')

    def _graph_dyn_fuse_plane(self, feat: torch.Tensor, bs: int, F_: int, metas, translation_mode: str, plane_key: str):
        if self.graph_dyn is None:
            return feat, feat.new_zeros(()), {}
        if plane_key not in self.graph_dyn_planes and plane_key not in self.graph_dyn_loss_planes:
            return feat, feat.new_zeros(()), {}

        c = int(feat.shape[1])
        h = int(feat.shape[2])
        w = int(feat.shape[3])
        feat_bf = feat.view(bs, F_, c, h, w)

        fused = []
        loss_total = feat.new_zeros(())
        loss_items = {}

        for b in range(bs):
            x = feat_bf[b].permute(0, 2, 3, 1).reshape(F_ * h * w, c)
            use_ego = bool(self.graph_dyn_use_ego_motion_compensation)
            if self.training and self.graph_dyn_ego_dropout > 0:
                if torch.rand((), device=x.device).item() < float(self.graph_dyn_ego_dropout):
                    use_ego = False
            g, stats = generate_adaptive_graph_optimized(
                int(F_), int(h), int(w), x,
                mode='training' if (self.training or self.graph_dyn_inference_use_dynamic_edges) else 'inference',
                similarity_threshold=0.05,
                spatial_radius=int(self.graph_dyn_spatial_radius),
                temporal_radius=int(self.graph_dyn_temporal_radius),
                k_additional=int(self.graph_dyn_k_additional),
                high_motion_top_ratio=float(self.graph_dyn_high_motion_top_ratio),
                verbose=False,
                timing_stats=None,
                metas=None if metas is None else (metas[b] if isinstance(metas, (list, tuple)) else metas),
                translation_mode=str(translation_mode),
                use_ego_motion_compensation=bool(use_ego),
            )

            if plane_key in self.graph_dyn_planes:
                z = self.graph_dyn.embed(g, x, H=int(h), W=int(w), frame=int(F_))
                z = self.graph_dyn_proj(z)
                z = z.view(F_, h, w, c).permute(0, 3, 1, 2).contiguous()
                if self.graph_dyn_fuse == 'replace':
                    fused.append(self.graph_dyn_scale * z)
                elif self.graph_dyn_fuse == 'add':
                    fused.append(feat_bf[b] + self.graph_dyn_scale * z)
                else:
                    fused.append(feat_bf[b] + self._graph_dyn_gate(plane_key) * z)
            else:
                fused.append(feat_bf[b])

            if self.training and plane_key in self.graph_dyn_loss_planes:
                loss, items = self.graph_dyn(g, x, H=int(h), W=int(w), frame=int(F_), metas=None if metas is None else (metas[b] if isinstance(metas, (list, tuple)) else metas), translation_mode=str(translation_mode))
                loss_total = loss_total + loss
                loss_items[f'{plane_key}_base_edges'] = float(stats.get('base_edges', 0))
                loss_items[f'{plane_key}_additional_edges'] = float(stats.get('additional_edges', 0))
                loss_items[f'{plane_key}_nodes'] = float(getattr(g, 'num_nodes', lambda: 0)())
                for k, v in (items or {}).items():
                    loss_items[f'{plane_key}_{k}'] = float(v)

        out = torch.stack(fused, dim=0).view(bs * F_, c, h, w)
        return out, loss_total, loss_items

    def forward_encoder(self, xxx, metas=None):
        bs, F, H, W, D = xxx.shape

        xxx = self.class_embeds(xxx)

        xxx = xxx.reshape(bs*F, H, W, D, self.expansion).permute(0, 4, 1, 2, 3)

        vol_feat = self.encoder(xxx)

        if self.pool_type == 'mean':
            xy_feat = vol_feat.mean(dim=4)
            xz_feat = vol_feat.mean(dim=3)
            yz_feat = vol_feat.mean(dim=2)
        else:
            logits = self.pool_logits(vol_feat)
            w_xy = torch.softmax(logits, dim=4)
            xy_feat = (vol_feat * w_xy).sum(dim=4)

            w_xz = torch.softmax(logits, dim=3)
            xz_feat = (vol_feat * w_xz).sum(dim=3)

            w_yz = torch.softmax(logits, dim=2)
            yz_feat = (vol_feat * w_yz).sum(dim=2)

        if self.temporal_dwconv is not None and F > 1:
            xy_feat = self._temporal_fuse_2d(xy_feat, bs, F)
            xz_feat = self._temporal_fuse_2d(xz_feat, bs, F)
            yz_feat = self._temporal_fuse_2d(yz_feat, bs, F)

        if self.partition_proj is not None:
            xy_feat = self.partition_proj(xy_feat)
            xz_feat = self.partition_proj(xz_feat)
            yz_feat = self.partition_proj(yz_feat)

        graph_loss = xy_feat.new_zeros(())
        graph_items = {}
        if self.graph_dyn is not None and F > 1:
            xy_s, xy_d = xy_feat[:, :self.static_channels], xy_feat[:, self.static_channels:]
            xz_s, xz_d = xz_feat[:, :self.static_channels], xz_feat[:, self.static_channels:]
            yz_s, yz_d = yz_feat[:, :self.static_channels], yz_feat[:, self.static_channels:]

            xy_d, l_xy, i_xy = self._graph_dyn_fuse_plane(xy_d, bs, F, metas, translation_mode='xy', plane_key='xy')
            xz_d, l_xz, i_xz = self._graph_dyn_fuse_plane(xz_d, bs, F, metas, translation_mode='xz', plane_key='xz')
            yz_d, l_yz, i_yz = self._graph_dyn_fuse_plane(yz_d, bs, F, metas, translation_mode='yz', plane_key='yz')

            xy_feat = torch.cat([xy_s, xy_d], dim=1)
            xz_feat = torch.cat([xz_s, xz_d], dim=1)
            yz_feat = torch.cat([yz_s, yz_d], dim=1)
            graph_loss = (l_xy + l_xz + l_yz)
            graph_items.update(i_xy)
            graph_items.update(i_xz)
            graph_items.update(i_yz)


        xy_feat = self._postprocess_plane(xy_feat)
        xz_feat = self._postprocess_plane(xz_feat)
        yz_feat = self._postprocess_plane(yz_feat)

        return [xy_feat, xz_feat, yz_feat], graph_loss, graph_items

    def _temporal_fuse_2d(self, feat: torch.Tensor, bs: int, F: int) -> torch.Tensor:
        c = int(feat.shape[1])
        h = int(feat.shape[2])
        w = int(feat.shape[3])
        feat = feat.view(bs, F, c, h, w).permute(0, 3, 4, 2, 1).contiguous()
        feat = feat.view(bs * h * w, c, F)
        feat = self.temporal_dwconv(feat)
        feat = feat.view(bs, h, w, c, F).permute(0, 4, 3, 1, 2).contiguous()
        return feat.view(bs * F, c, h, w)

    def _postprocess_plane(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.use_static_dynamic_split or self.dynamic_channels == 0:
            x = self.norm_static(feat)
            if self.plane_act == 'tanh':
                x = (x * self.plane_act_scale).tanh()
            elif self.plane_act == 'none':
                x = x
            elif self.plane_act == 'clamp':
                x = torch.clamp(x * self.plane_act_scale, -1.0, 1.0)
            else:
                raise ValueError(f'unsupported plane_act: {self.plane_act}')
            return x

        s = feat[:, :self.static_channels]
        d = feat[:, self.static_channels:]

        s = self.norm_static(s)
        if self.plane_act == 'tanh':
            s = (s * self.plane_act_scale).tanh()
        elif self.plane_act == 'none':
            s = s
        elif self.plane_act == 'clamp':
            s = torch.clamp(s * self.plane_act_scale, -1.0, 1.0)
        else:
            raise ValueError(f'unsupported plane_act: {self.plane_act}')

        d = self.norm_dynamic(d)
        d = d * self.dynamic_scale
        if self.dynamic_act == 'tanh':
            d = (d * self.dynamic_act_scale).tanh()
        elif self.dynamic_act == 'none':
            d = d * self.dynamic_act_scale
        elif self.dynamic_act == 'clamp':
            d = torch.clamp(d * self.dynamic_act_scale, -1.0, 1.0)
        else:
            raise ValueError(f'unsupported dynamic_act: {self.dynamic_act}')

        return torch.cat([s, d], dim=1)

    def sample_feature_plane2D(self, feat_map, x):
        sample_coords = x.view(x.shape[0], 1, -1, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1), align_corners=True, padding_mode='border')
        feat = feat[:, :, 0, :]
        feat = feat.transpose(1, 2)
        return feat

    def forward_decoder(self, feat_maps, querys):

        coords_list = [[0, 1], [0, 2], [1, 2]]
        geo_feat_maps = [fm[:, :self.expansion] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        h_geo = 0
        for i in range(3):
            h_geo += self.sample_feature_plane2D(
                geo_feat_maps[i], querys[..., coords_list[i]])

        PE = []
        for freq in range(self.pos_num_freq):
            PE.append(torch.sin((2.**freq) * querys))
            PE.append(torch.cos((2.**freq) * querys))

        PE = torch.cat(PE, dim=-1)
        h_geo = torch.cat([h_geo, PE], dim=-1)

        preds = self.decoder(h_geo)

        return preds

    def forward(self, x, querys, xyz_labels, xyz_centers, **kwargs):


        output_dict = {}
        metas = kwargs.get('metas', None)
        feat_maps, graph_loss, graph_items = self.forward_encoder(x, metas=metas)

        if self.only_triplane:
            return feat_maps

        output_dict['graph_dyn_loss'] = graph_loss
        if graph_items:
            output_dict['graph_dyn_items'] = graph_items

        preds = self.forward_decoder(feat_maps, querys)

        softmax_preds = torch.nn.functional.softmax(preds, dim=2)

        empty_label = 0.
        pred_logits = torch.full(
            (preds.shape[0], 200, 200, 16, self.num_cls),
            fill_value=empty_label, device=preds.device)
        pred_output = torch.full(
            (preds.shape[0], 200, 200, 16, self.num_cls),
            fill_value=empty_label, device=preds.device)
        pred_logits[:, :, :, :, -1] = 1
        pred_output[:, :, :, :, -1] = 1
        for i in range(softmax_preds.shape[0]):
            pred_output[i,
                xyz_centers[i, :, 0],
                xyz_centers[i, :, 1],
                xyz_centers[i, :, 2], :] = softmax_preds[i]
            pred_logits[i,
                xyz_centers[i, :, 0],
                xyz_centers[i, :, 1],
                xyz_centers[i, :, 2], :] = preds[i]

        output_dict.update({'preds': preds, 'pred_output': pred_output})

        output_dict['feat_maps'] = feat_maps

        if not self.training:
            pred = pred_output.unsqueeze(0).argmax(dim=-1).detach().cuda()
            output_dict['sem_pred'] = pred
            pred_iou = deepcopy(pred)

            pred_iou[pred_iou!=17] = 1
            pred_iou[pred_iou==17] = 0
            output_dict['iou_pred'] = pred_iou


        return output_dict

    def generate(self, z, shapes, input_shape):
        logits = self.forward_decoder(z, shapes, input_shape)
        return {'logits': logits}


@MODELS.register_module()
class TripLaneEncoder(BaseModule):
    def __init__(self, geo_feat_channels=8, z_down=True, all_down = False, padding_mode="replicate",
                        kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.all_down = all_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )


        if self.all_down:
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(4, 4, 2), stride=(4, 4, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        elif self.z_down:
                        self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )


        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )


    def forward(self, x):
        x = self.conv0(x)

        residual_feat = x
        x = self.convblock1(x)
        x = x + residual_feat
        x = self.downsample(x)

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return x


class TriplaneGroupResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False, ks=3, input_norm=True, input_act=True):
        super().__init__()
        in_channels *= 3
        out_channels *= 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up

        self.input_norm = input_norm
        if input_norm and input_act:
            self.in_layers = nn.Sequential(
                SiLU(),
                nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            )
        elif not input_norm:
            if input_act:
                self.in_layers = nn.Sequential(
                    SiLU(),
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
            else:
                self.in_layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
                )
        else:
            raise NotImplementedError

        self.norm_xy = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_xz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)
        self.norm_yz = nn.InstanceNorm2d(out_channels//3, eps=1e-6, affine=True)

        self.out_layers = nn.Sequential(
            SiLU(),
            zero_module(
                nn.Conv2d(out_channels, out_channels, groups=3, kernel_size=ks, stride=1, padding=(ks - 1)//2)
            ),
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, groups=3, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, feat_maps):
        if self.input_norm:
            feat_maps = [self.norm_xy(feat_maps[0]), self.norm_xz(feat_maps[1]), self.norm_yz(feat_maps[2])]
        x, (H, W, D) = compose_triplane_channelwise(feat_maps)

        if self.up:
            raise NotImplementedError
        else:
            h = self.in_layers(x)

        h_xy, h_xz, h_yz = decompose_triplane_channelwise(h, (H, W, D))
        h_xy = self.norm_xy(h_xy)
        h_xz = self.norm_xz(h_xz)
        h_yz = self.norm_yz(h_yz)
        h, _ = compose_triplane_channelwise([h_xy, h_xz, h_yz])

        h = self.out_layers(h)
        h = h + self.shortcut(x)
        h_maps = decompose_triplane_channelwise(h, (H, W, D))
        return h_maps


@MODELS.register_module()
class TripLaneDecoder(BaseModule):
    def __init__(self, in_channels=100, out_channels=18,
                        hidden_channels=256, num_hidden_layers=4, posenc=0):
        super().__init__()
        self.posenc = posenc
        if posenc > 0:
            self.PE = SinusoidalEncoder(in_channels, 0, posenc, use_identity=True)
            in_channels = self.PE.latent_dim
        first_layer_list = [nn.Linear(in_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2):
            first_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            first_layer_list.append(nn.ReLU())
        self.first_layers = nn.Sequential(*first_layer_list)

        second_layer_list = [nn.Linear(in_channels + hidden_channels, hidden_channels), nn.ReLU()]
        for _ in range(num_hidden_layers // 2 - 1):
            second_layer_list.append(nn.Linear(hidden_channels, hidden_channels))
            second_layer_list.append(nn.ReLU())
        second_layer_list.append(nn.Linear(hidden_channels, out_channels))
        self.second_layers = nn.Sequential(*second_layer_list)

    def forward(self, x):
        if self.posenc > 0:
            x = self.PE(x)
        h = self.first_layers(x)
        h = torch.cat([x, h], dim=-1)
        h = self.second_layers(h)
        return h
