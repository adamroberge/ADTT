from copy import deepcopy

from mmengine.model import BaseModule
from mmengine.registry import MODELS

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .TripLane.TripLane import TriplaneGroupResnetBlock


def build_2d_sincos_pos_embed(height, width, dim, device):
    if int(dim) <= 0:
        raise ValueError(f'positional dim must be positive, got {dim}')
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, device=device),
        torch.linspace(-1.0, 1.0, steps=width, device=device),
        indexing='ij',
    )
    bands = max(1, math.ceil(int(dim) / 4))
    omega = torch.arange(bands, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, bands)))

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    pos = pos[:, :int(dim)]
    return pos


def build_3d_sincos_pos_embed(size_x, size_y, size_z, dim, device):
    if int(dim) <= 0:
        raise ValueError(f'3d positional dim must be positive, got {dim}')
    x, y, z = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=size_x, device=device),
        torch.linspace(-1.0, 1.0, steps=size_y, device=device),
        torch.linspace(-1.0, 1.0, steps=size_z, device=device),
        indexing='ij',
    )
    bands = max(1, math.ceil(int(dim) / 6))
    omega = torch.arange(bands, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, bands)))

    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    z = z.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat(
        [torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y), torch.sin(z), torch.cos(z)],
        dim=1,
    )
    return pos[:, :int(dim)]


def quat_wxyz_to_rotmat(quat):
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quat_wxyz_to_yaw(quat):
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _group_norm_groups(num_channels, max_groups=8):
    groups = max(1, min(int(max_groups), int(num_channels)))
    while int(num_channels) % groups != 0 and groups > 1:
        groups -= 1
    return groups


class PlaneTokenizer(nn.Module):
    def __init__(self, in_channels, token_dim, pooled_hw, plane_name):
        super().__init__()
        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.pooled_hw = tuple(int(v) for v in pooled_hw)
        groups = _group_norm_groups(self.in_channels)
        self.pre_norm = nn.GroupNorm(groups, self.in_channels)
        self.local_refine = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, groups=self.in_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
        )
        self.pool_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.plane_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.input_proj = nn.Linear(self.in_channels, self.token_dim)
        self.output_proj = nn.Linear(self.token_dim, self.in_channels)
        self.norm = nn.LayerNorm(self.token_dim)
        self.plane_name = str(plane_name)

    @property
    def num_tokens(self):
        return int(self.pooled_hw[0] * self.pooled_hw[1])

    def encode(self, plane):
        refined = plane + self.local_refine(self.pre_norm(plane))
        pooled_avg = F.adaptive_avg_pool2d(refined, self.pooled_hw)
        pooled_max = F.adaptive_max_pool2d(refined, self.pooled_hw)
        gate = torch.sigmoid(self.pool_gate)
        pooled = pooled_avg * (1.0 - gate) + pooled_max * gate
        tokens = pooled.flatten(2).transpose(1, 2)
        pos = build_2d_sincos_pos_embed(self.pooled_hw[0], self.pooled_hw[1], self.token_dim, plane.device)
        tokens = self.input_proj(tokens) + pos.unsqueeze(0) + self.plane_embed
        return self.norm(tokens)

    def decode(self, tokens, out_hw):
        plane = self.output_proj(tokens)
        plane = plane.transpose(1, 2).reshape(tokens.shape[0], self.in_channels, self.pooled_hw[0], self.pooled_hw[1])
        if tuple(out_hw) != self.pooled_hw:
            plane = F.interpolate(plane, size=tuple(out_hw), mode='bilinear', align_corners=False)
        return plane


class TriplanePointTokenizer3D(nn.Module):
    def __init__(self, in_channels, token_dim, grid_size):
        super().__init__()
        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.grid_size = tuple(int(v) for v in grid_size)
        if len(self.grid_size) != 3:
            raise ValueError(f'grid_size must have 3 dims, got {self.grid_size}')
        groups = _group_norm_groups(self.in_channels)

        def make_refine():
            return nn.Sequential(
                nn.GroupNorm(groups, self.in_channels),
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.in_channels,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            )

        self.xy_refine = make_refine()
        self.xz_refine = make_refine()
        self.yz_refine = make_refine()
        self.xy_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.xz_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.yz_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.fuse_proj = nn.Sequential(
            nn.Conv3d(self.in_channels * 3, self.in_channels * 2, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=False),
        )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, self.token_dim),
        )
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_embed = nn.Parameter(torch.zeros(1, 1, self.token_dim))

    @property
    def num_tokens(self):
        return int(self.grid_size[0] * self.grid_size[1] * self.grid_size[2])

    def _downsample_plane(self, plane, refine, pooled_hw, gate_param):
        refined = refine(plane)
        pooled_avg = F.adaptive_avg_pool2d(refined, pooled_hw)
        pooled_max = F.adaptive_max_pool2d(refined, pooled_hw)
        gate = torch.sigmoid(gate_param)
        return pooled_avg * (1.0 - gate) + pooled_max * gate

    def _coords(self, device, dtype):
        x_bins, y_bins, z_bins = self.grid_size
        x = torch.linspace(-40.0 + 40.0 / x_bins, 40.0 - 40.0 / x_bins, steps=x_bins, device=device, dtype=dtype)
        y = torch.linspace(-40.0 + 40.0 / y_bins, 40.0 - 40.0 / y_bins, steps=y_bins, device=device, dtype=dtype)
        z = torch.linspace(-1.0 + 3.2 / z_bins, 5.4 - 3.2 / z_bins, steps=z_bins, device=device, dtype=dtype)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    def encode(self, xy_plane, xz_plane, yz_plane):
        x_bins, y_bins, z_bins = self.grid_size
        xy_feat = self._downsample_plane(xy_plane, self.xy_refine, (x_bins, y_bins), self.xy_gate)
        xz_feat = self._downsample_plane(xz_plane, self.xz_refine, (x_bins, z_bins), self.xz_gate)
        yz_feat = self._downsample_plane(yz_plane, self.yz_refine, (y_bins, z_bins), self.yz_gate)

        xy_feat = xy_feat.unsqueeze(4).expand(-1, -1, x_bins, y_bins, z_bins)
        xz_feat = xz_feat.unsqueeze(3).expand(-1, -1, x_bins, y_bins, z_bins)
        yz_feat = yz_feat.unsqueeze(2).expand(-1, -1, x_bins, y_bins, z_bins)
        fused = self.fuse_proj(torch.cat([xy_feat, xz_feat, yz_feat], dim=1))
        tokens = fused.permute(0, 2, 3, 4, 1).reshape(xy_plane.shape[0], self.num_tokens, self.in_channels)
        tokens = self.input_proj(tokens)
        pos = build_3d_sincos_pos_embed(x_bins, y_bins, z_bins, self.token_dim, xy_plane.device)
        tokens = self.token_norm(tokens + pos.unsqueeze(0) + self.token_embed)
        coords = self._coords(xy_plane.device, xy_plane.dtype)
        return tokens, coords


class DetrDecoderLayerWithAttn(nn.Module):
    def __init__(self, token_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            int(token_dim), int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            int(token_dim), int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.linear1 = nn.Linear(int(token_dim), int(dim_feedforward))
        self.dropout = nn.Dropout(float(dropout))
        self.linear2 = nn.Linear(int(dim_feedforward), int(token_dim))
        self.norm1 = nn.LayerNorm(int(token_dim))
        self.norm2 = nn.LayerNorm(int(token_dim))
        self.norm3 = nn.LayerNorm(int(token_dim))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.dropout3 = nn.Dropout(float(dropout))
        self.activation = nn.GELU()

    def forward(self, tgt, memory):
        q = self.norm1(tgt)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        tgt = tgt + self.dropout1(self_out)

        q = self.norm2(tgt)
        cross_out, cross_attn = self.cross_attn(
            q,
            memory,
            memory,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout2(cross_out)

        ff = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout3(ff)
        return tgt, cross_attn


class DetrStyleTriplaneTransformer(nn.Module):
    def __init__(
        self,
        token_dim,
        num_xy_tokens,
        num_xz_tokens,
        num_yz_tokens,
        depth=4,
        num_heads=4,
        dropout=0.1,
        mlp_ratio=4.0,
        max_time_steps=64,
        max_future_steps=32,
    ):
        super().__init__()
        token_dim = int(token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(token_dim * mlp_ratio),
            dropout=float(dropout),
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.decoder_layers = nn.ModuleList([
            DetrDecoderLayerWithAttn(
                token_dim=token_dim,
                num_heads=num_heads,
                dim_feedforward=int(token_dim * mlp_ratio),
                dropout=dropout,
            )
            for _ in range(int(depth))
        ])
        self.memory_norm = nn.LayerNorm(token_dim)
        self.decoder_norm = nn.LayerNorm(token_dim)

        self.num_xy_tokens = int(num_xy_tokens)
        self.num_xz_tokens = int(num_xz_tokens)
        self.num_yz_tokens = int(num_yz_tokens)
        self.history_time_embed = nn.Parameter(torch.zeros(1, int(max_time_steps), 1, token_dim))
        self.future_step_embed = nn.Parameter(torch.zeros(1, int(max_future_steps), 1, token_dim))
        self.delta_query_xy = nn.Parameter(torch.randn(1, self.num_xy_tokens, token_dim) * 0.02)
        self.delta_query_xz = nn.Parameter(torch.randn(1, self.num_xz_tokens, token_dim) * 0.02)
        self.delta_query_yz = nn.Parameter(torch.randn(1, self.num_yz_tokens, token_dim) * 0.02)
        self.known_query_xy = nn.Parameter(torch.randn(1, self.num_xy_tokens, token_dim) * 0.02)
        self.known_query_xz = nn.Parameter(torch.randn(1, self.num_xz_tokens, token_dim) * 0.02)
        self.known_query_yz = nn.Parameter(torch.randn(1, self.num_yz_tokens, token_dim) * 0.02)
        self.unknown_query_xy = nn.Parameter(torch.randn(1, self.num_xy_tokens, token_dim) * 0.02)
        self.unknown_query_xz = nn.Parameter(torch.randn(1, self.num_xz_tokens, token_dim) * 0.02)
        self.unknown_query_yz = nn.Parameter(torch.randn(1, self.num_yz_tokens, token_dim) * 0.02)

    def _time_embed(self, length, device, dtype):
        if length > self.history_time_embed.shape[1]:
            raise ValueError(
                f'history length {length} exceeds max_time_steps={self.history_time_embed.shape[1]}'
            )
        return self.history_time_embed[:, :length].to(device=device, dtype=dtype)

    def _step_embed(self, step_idx, device, dtype):
        step_idx = min(int(step_idx), int(self.future_step_embed.shape[1]) - 1)
        return self.future_step_embed[:, step_idx:step_idx + 1].to(device=device, dtype=dtype)

    def encode_memory(self, xy_tokens, xz_tokens, yz_tokens):
        batch_size, history_len = xy_tokens.shape[:2]
        time_embed = self._time_embed(history_len, xy_tokens.device, xy_tokens.dtype)
        xy_tokens = xy_tokens + time_embed
        xz_tokens = xz_tokens + time_embed
        yz_tokens = yz_tokens + time_embed
        memory = torch.cat([xy_tokens, xz_tokens, yz_tokens], dim=2)
        memory = memory.reshape(batch_size, -1, memory.shape[-1])
        return self.memory_norm(self.encoder(memory))

    def encode_memory_tokens(self, memory_tokens):
        batch_size, history_len = memory_tokens.shape[:2]
        time_embed = self._time_embed(history_len, memory_tokens.device, memory_tokens.dtype)
        memory_tokens = memory_tokens + time_embed
        memory_tokens = memory_tokens.reshape(batch_size, -1, memory_tokens.shape[-1])
        return self.memory_norm(self.encoder(memory_tokens))

    def decode_delta(self, memory, ego_token, external_tokens, step_idx):
        batch_size = int(memory.shape[0])
        step_embed = self._step_embed(step_idx, memory.device, memory.dtype)
        control_tokens = torch.cat([ego_token, external_tokens], dim=1) + step_embed.squeeze(1)
        delta_queries = torch.cat([
            self.delta_query_xy.expand(batch_size, -1, -1),
            self.delta_query_xz.expand(batch_size, -1, -1),
            self.delta_query_yz.expand(batch_size, -1, -1),
        ], dim=1) + step_embed.squeeze(1)
        tgt = torch.cat([control_tokens, delta_queries], dim=1)
        cross_attn = None
        decoded = tgt
        for decoder_layer in self.decoder_layers:
            decoded, cross_attn = decoder_layer(decoded, memory)
        decoded = self.decoder_norm(decoded)

        control_len = int(control_tokens.shape[1])
        delta_tokens = decoded[:, control_len:]
        control_cross_attn = cross_attn.mean(dim=1)[:, :control_len] if cross_attn is not None else None
        xy_end = self.num_xy_tokens
        xz_end = xy_end + self.num_xz_tokens
        return (
            delta_tokens[:, :xy_end],
            delta_tokens[:, xy_end:xz_end],
            delta_tokens[:, xz_end:],
            decoded[:, :control_len],
            control_cross_attn,
        )

    def decode_control_delta(self, memory, ego_token, external_tokens, step_idx):
        step_embed = self._step_embed(step_idx, memory.device, memory.dtype)
        control_tokens = torch.cat([ego_token, external_tokens], dim=1) + step_embed.squeeze(1)
        expected_tokens = self.num_xy_tokens + self.num_xz_tokens + self.num_yz_tokens
        if int(control_tokens.shape[1]) != expected_tokens:
            raise ValueError(
                f'control query count must match triplane token count, got '
                f'{control_tokens.shape[1]} vs {expected_tokens}'
            )

        cross_attn = None
        decoded = control_tokens
        for decoder_layer in self.decoder_layers:
            decoded, cross_attn = decoder_layer(decoded, memory)
        decoded = self.decoder_norm(decoded)

        control_cross_attn = cross_attn.mean(dim=1) if cross_attn is not None else None
        xy_end = self.num_xy_tokens
        xz_end = xy_end + self.num_xz_tokens
        return (
            decoded[:, :xy_end],
            decoded[:, xy_end:xz_end],
            decoded[:, xz_end:],
            decoded,
            control_cross_attn,
        )

    def decode_query_tokens(self, memory, query_tokens, step_idx):
        step_embed = self._step_embed(step_idx, memory.device, memory.dtype)
        control_tokens = query_tokens + step_embed.squeeze(1)
        expected_tokens = self.num_xy_tokens + self.num_xz_tokens + self.num_yz_tokens
        if int(control_tokens.shape[1]) != expected_tokens:
            raise ValueError(
                f'query token count must match triplane token count, got '
                f'{int(control_tokens.shape[1])} vs {expected_tokens}'
            )

        cross_attn = None
        decoded = control_tokens
        for decoder_layer in self.decoder_layers:
            decoded, cross_attn = decoder_layer(decoded, memory)
        decoded = self.decoder_norm(decoded)

        control_cross_attn = cross_attn.mean(dim=1) if cross_attn is not None else None
        xy_end = self.num_xy_tokens
        xz_end = xy_end + self.num_xz_tokens
        return (
            decoded[:, :xy_end],
            decoded[:, xy_end:xz_end],
            decoded[:, xz_end:],
            decoded,
            control_cross_attn,
        )

    def decode_latent_queries(self, memory, query_tokens, step_idx):
        step_embed = self._step_embed(step_idx, memory.device, memory.dtype)
        decoded = query_tokens + step_embed.squeeze(1)
        cross_attn = None
        for decoder_layer in self.decoder_layers:
            decoded, cross_attn = decoder_layer(decoded, memory)
        decoded = self.decoder_norm(decoded)
        control_cross_attn = cross_attn.mean(dim=1) if cross_attn is not None else None
        return decoded, control_cross_attn

    def decode_split_control_delta(self, memory, ego_token, external_tokens, step_idx, known_query_mask):
        step_embed = self._step_embed(step_idx, memory.device, memory.dtype).squeeze(1)
        batch_size = int(memory.shape[0])
        total_tokens = self.num_xy_tokens + self.num_xz_tokens + self.num_yz_tokens
        if tuple(known_query_mask.shape[:2]) != (batch_size, total_tokens):
            raise ValueError(
                f'known_query_mask shape must be [B, {total_tokens}], got {tuple(known_query_mask.shape)}'
            )

        known_pos = torch.cat([
            self.known_query_xy.expand(batch_size, -1, -1),
            self.known_query_xz.expand(batch_size, -1, -1),
            self.known_query_yz.expand(batch_size, -1, -1),
        ], dim=1)
        unknown_pos = torch.cat([
            self.unknown_query_xy.expand(batch_size, -1, -1),
            self.unknown_query_xz.expand(batch_size, -1, -1),
            self.unknown_query_yz.expand(batch_size, -1, -1),
        ], dim=1)
        ego_queries = known_pos + ego_token.expand(-1, total_tokens, -1) + step_embed
        ext_summary = external_tokens.mean(dim=1, keepdim=True)
        unknown_queries = unknown_pos + ext_summary.expand(-1, total_tokens, -1) + step_embed

        known_decoded = ego_queries
        unknown_decoded = unknown_queries
        known_cross_attn = None
        unknown_cross_attn = None
        for decoder_layer in self.decoder_layers:
            known_decoded, known_cross_attn = decoder_layer(known_decoded, memory)
            unknown_decoded, unknown_cross_attn = decoder_layer(unknown_decoded, memory)
        known_decoded = self.decoder_norm(known_decoded)
        unknown_decoded = self.decoder_norm(unknown_decoded)

        known_mask = known_query_mask.unsqueeze(-1).to(dtype=memory.dtype)
        unknown_mask = 1.0 - known_mask
        merged_decoded = known_decoded * known_mask + unknown_decoded * unknown_mask
        merged_cross_attn = None
        if known_cross_attn is not None and unknown_cross_attn is not None:
            known_mask_attn = known_query_mask.unsqueeze(1).unsqueeze(-1).to(dtype=memory.dtype)
            unknown_mask_attn = 1.0 - known_mask_attn
            merged_cross_attn = known_cross_attn * known_mask_attn + unknown_cross_attn * unknown_mask_attn
            merged_cross_attn = merged_cross_attn.mean(dim=1)

        xy_end = self.num_xy_tokens
        xz_end = xy_end + self.num_xz_tokens
        return {
            'known_xy': known_decoded[:, :xy_end],
            'known_xz': known_decoded[:, xy_end:xz_end],
            'known_yz': known_decoded[:, xz_end:],
            'unknown_xy': unknown_decoded[:, :xy_end],
            'unknown_xz': unknown_decoded[:, xy_end:xz_end],
            'unknown_yz': unknown_decoded[:, xz_end:],
            'merged_xy': merged_decoded[:, :xy_end],
            'merged_xz': merged_decoded[:, xy_end:xz_end],
            'merged_yz': merged_decoded[:, xz_end:],
            'merged_queries': merged_decoded,
            'known_queries': known_decoded,
            'unknown_queries': unknown_decoded,
            'cross_attn': merged_cross_attn,
        }


class ControlCrossAttention(nn.Module):
    def __init__(self, token_dim, num_heads, dropout=0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(int(token_dim))
        self.mem_norm = nn.LayerNorm(int(token_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(token_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(int(token_dim))

    def forward(self, query_tokens, memory_tokens):
        q = self.query_norm(query_tokens)
        kv = self.mem_norm(memory_tokens)
        out, attn = self.attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        out = self.out_norm(out + query_tokens)
        return out, attn


class ControlRegionProjector(nn.Module):
    def __init__(self, token_dim, num_query_tokens):
        super().__init__()
        self.num_query_tokens = int(num_query_tokens)
        self.region_head = nn.Linear(int(token_dim), self.num_query_tokens)
        self.value_proj = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
        )

    def forward(self, control_ctx):
        region_logits = self.region_head(control_ctx)
        region_weights = torch.softmax(region_logits, dim=-1)
        value_tokens = self.value_proj(control_ctx)
        query_tokens = (region_weights.unsqueeze(-1) * value_tokens.unsqueeze(2)).sum(dim=1)
        return query_tokens, region_weights


class PlaneDeltaDecoder(nn.Module):
    def __init__(self, token_dim, plane_channels):
        super().__init__()
        self.token_refine = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim)),
            nn.GELU(),
            nn.Linear(int(token_dim), int(token_dim)),
        )
        self.delta_gate = nn.Sequential(
            nn.Linear(int(token_dim), int(token_dim)),
            nn.GELU(),
            nn.Linear(int(token_dim), 1),
            nn.Sigmoid(),
        )
        self.plane_channels = int(plane_channels)

    def forward(self, latest_tokens, delta_tokens):
        refined = self.token_refine(delta_tokens)
        gate = self.delta_gate(refined)
        return latest_tokens + refined * gate


class GatedResidualFusionBlock(nn.Module):
    def __init__(self, plane_channels):
        super().__init__()
        c = int(plane_channels)
        groups = _group_norm_groups(c)
        self.offset = nn.Conv2d(c * 3, 18, kernel_size=3, padding=1)
        self.deform = DeformConv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.post = nn.Sequential(
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, target, other_a, other_b):
        ctx = torch.cat([target, other_a, other_b], dim=1)
        residual = self.post(self.deform(target, self.offset(ctx)))
        gate = self.gate(ctx)
        return target + gate * residual


class GatedResidualDeformFusion(nn.Module):
    def __init__(self, plane_channels, num_blocks=2):
        super().__init__()
        c = int(plane_channels)
        groups = _group_norm_groups(c)
        self.xy_from_xz = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.xy_from_yz = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.xz_from_xy = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.xz_from_yz = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.yz_from_xy = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.yz_from_xz = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.xy_blocks = nn.ModuleList([GatedResidualFusionBlock(c) for _ in range(int(num_blocks))])
        self.xz_blocks = nn.ModuleList([GatedResidualFusionBlock(c) for _ in range(int(num_blocks))])
        self.yz_blocks = nn.ModuleList([GatedResidualFusionBlock(c) for _ in range(int(num_blocks))])
        self.out_xy = nn.Sequential(
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )
        self.out_xz = nn.Sequential(
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )
        self.out_yz = nn.Sequential(
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def _resize(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, xy, xz, yz):
        xy_other_xz = self.xy_from_xz(self._resize(xz, xy.shape[-2:]))
        xy_other_yz = self.xy_from_yz(self._resize(yz, xy.shape[-2:]))
        xz_other_xy = self.xz_from_xy(self._resize(xy, xz.shape[-2:]))
        xz_other_yz = self.xz_from_yz(self._resize(yz, xz.shape[-2:]))
        yz_other_xy = self.yz_from_xy(self._resize(xy, yz.shape[-2:]))
        yz_other_xz = self.yz_from_xz(self._resize(xz, yz.shape[-2:]))

        for block in self.xy_blocks:
            xy = block(xy, xy_other_xz, xy_other_yz)
        for block in self.xz_blocks:
            xz = block(xz, xz_other_xy, xz_other_yz)
        for block in self.yz_blocks:
            yz = block(yz, yz_other_xy, yz_other_xz)

        xy = xy + self.out_xy(xy)
        xz = xz + self.out_xz(xz)
        yz = yz + self.out_yz(yz)
        return xy, xz, yz


class LowRes3DToTriplaneProjector(nn.Module):
    def __init__(self, token_dim, plane_channels):
        super().__init__()
        c = int(token_dim)
        p = int(plane_channels)
        groups = _group_norm_groups(c)
        self.volume_refine = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
        )
        self.xy_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, p, kernel_size=1, bias=True),
        )
        self.xz_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, p, kernel_size=1, bias=True),
        )
        self.yz_proj = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, p, kernel_size=1, bias=True),
        )

    def forward(self, latent_tokens, grid_size, out_shapes):
        batch_size, num_tokens, channels = latent_tokens.shape
        gx, gy, gz = [int(v) for v in grid_size]
        if num_tokens != gx * gy * gz:
            raise ValueError(
                f'latent token count {num_tokens} does not match grid size {grid_size}'
            )
        volume = latent_tokens.transpose(1, 2).reshape(batch_size, channels, gx, gy, gz)
        volume = volume + self.volume_refine(volume)

        x_size = int(out_shapes['xy'][0])
        y_size = int(out_shapes['xy'][1])
        z_size = int(out_shapes['xz'][1])
        if int(out_shapes['xz'][0]) != x_size or int(out_shapes['yz'][0]) != y_size or int(out_shapes['yz'][1]) != z_size:
            raise ValueError(
                f'inconsistent triplane output shapes: {out_shapes}'
            )
        volume = F.interpolate(volume, size=(x_size, y_size, z_size), mode='nearest')
        xy = self.xy_proj(volume.mean(dim=4))
        xz = self.xz_proj(volume.mean(dim=3))
        yz = self.yz_proj(volume.mean(dim=2))
        return xy, xz, yz


@MODELS.register_module()
class DeltaTokenPredictor(BaseModule):
    def __init__(
        self,
        prev_steps=4,
        num_frames=9,
        plane_channels=2,
        token_dim=128,
        context_frames=8,
        transformer_depth=4,
        num_heads=4,
        dropout=0.1,
        xy_hw=(8, 8),
        xz_hw=(8, 4),
        yz_hw=(8, 4),
        control_dim=30,
        num_real_control_tokens=1,
        num_learnable_control_tokens=4,
        triplane_encoder_cfg=None,
        triplane_decoder_cfg=None,
        num_classes=18,
        control_query_only_decoder=False,
        use_motion_compensation=True,
        predict_delta=True,
        ego_only_control_queries=False,
        warp_sample_mode='nearest',
        warp_padding_mode='zeros',
        decoder_query_count=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.prev_steps = int(prev_steps)
        self.num_frames = int(num_frames)
        self.plane_channels = int(plane_channels)
        self.token_dim = int(token_dim)
        self.context_frames = int(context_frames)
        self.control_dim = int(control_dim)
        self.num_real_control_tokens = int(num_real_control_tokens)
        self.num_learnable_control_tokens = int(num_learnable_control_tokens)
        self.num_cls = int(num_classes)
        self.joint_token_dim = int(self.token_dim)
        self.control_query_only_decoder = bool(control_query_only_decoder)
        self.use_motion_compensation = bool(use_motion_compensation)
        self.predict_delta = bool(predict_delta)
        self.ego_only_control_queries = bool(ego_only_control_queries)
        self.warp_sample_mode = str(warp_sample_mode)
        self.warp_padding_mode = str(warp_padding_mode)
        if self.warp_sample_mode not in ['nearest', 'bilinear']:
            raise ValueError(f'unsupported warp_sample_mode: {self.warp_sample_mode}')
        if self.warp_padding_mode not in ['zeros', 'border', 'reflection']:
            raise ValueError(f'unsupported warp_padding_mode: {self.warp_padding_mode}')
        if triplane_encoder_cfg is None:
            raise ValueError('triplane_encoder_cfg is required for DeltaTokenPredictor')
        self.triplane_encoder = MODELS.build(triplane_encoder_cfg)
        self.triplane_class_embeds = nn.Embedding(self.num_cls, self.plane_channels)
        self.triplane_encoder_norm = nn.InstanceNorm2d(self.plane_channels)
        for module in [self.triplane_encoder, self.triplane_class_embeds, self.triplane_encoder_norm]:
            for param in module.parameters():
                param.requires_grad = False

        self.tokenizer_xy = PlaneTokenizer(self.plane_channels, self.token_dim, xy_hw, 'xy')
        self.tokenizer_xz = PlaneTokenizer(self.plane_channels, self.token_dim, xz_hw, 'xz')
        self.tokenizer_yz = PlaneTokenizer(self.plane_channels, self.token_dim, yz_hw, 'yz')
        self.total_triplane_tokens = (
            self.tokenizer_xy.num_tokens + self.tokenizer_xz.num_tokens + self.tokenizer_yz.num_tokens
        )
        self.decoder_query_count = int(decoder_query_count) if decoder_query_count is not None else self.total_triplane_tokens
        if self.control_query_only_decoder and (not self.ego_only_control_queries):
            if self.num_real_control_tokens <= 0:
                raise ValueError(f'num_real_control_tokens must be positive, got {self.num_real_control_tokens}')
            expected_learnable_tokens = self.decoder_query_count - self.num_real_control_tokens
            if expected_learnable_tokens < 0:
                raise ValueError(
                    f'num_real_control_tokens={self.num_real_control_tokens} exceeds decoder_query_count={self.decoder_query_count}'
                )
            if self.num_learnable_control_tokens <= 0:
                self.num_learnable_control_tokens = expected_learnable_tokens
            elif self.num_learnable_control_tokens != expected_learnable_tokens:
                raise ValueError(
                    'DeltaTokenPredictorV2 requires real control tokens + learnable control tokens '
                    f'to equal total triplane tokens. Expected num_learnable_control_tokens='
                    f'{expected_learnable_tokens}, got {self.num_learnable_control_tokens}. '
                    f'(decoder_query_count={self.decoder_query_count}, real={self.num_real_control_tokens})'
                )

        self.control_token_mlp = nn.Sequential(
            nn.Linear(self.control_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.real_control_adapter = None
        if self.num_real_control_tokens > 1:
            self.real_control_adapter = nn.Sequential(
                nn.LayerNorm(self.joint_token_dim),
                nn.Linear(self.joint_token_dim, self.joint_token_dim),
                nn.GELU(),
                nn.Linear(self.joint_token_dim, self.num_real_control_tokens * self.joint_token_dim),
            )
        self.learnable_control_queries = nn.Parameter(
            torch.randn(1, self.num_learnable_control_tokens, self.token_dim) * 0.02
        )
        self.learnable_control_adapter = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.num_learnable_control_tokens * self.token_dim),
        )
        self.control_query_norm = nn.LayerNorm(self.token_dim)
        self.control_to_joint = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.joint_token_dim),
        )
        self.external_control_queries = nn.Parameter(
            torch.randn(1, self.num_learnable_control_tokens, self.joint_token_dim) * 0.02
        )
        self.external_control_norm = nn.LayerNorm(self.joint_token_dim)
        self.triplane_transformer = DetrStyleTriplaneTransformer(
            token_dim=self.joint_token_dim,
            num_xy_tokens=self.tokenizer_xy.num_tokens,
            num_xz_tokens=self.tokenizer_xz.num_tokens,
            num_yz_tokens=self.tokenizer_yz.num_tokens,
            depth=transformer_depth,
            num_heads=num_heads,
            dropout=dropout,
            max_time_steps=max(64, self.prev_steps + self.num_frames + self.context_frames + 4),
            max_future_steps=max(32, self.num_frames + 4),
        )
        self.ego_query_adapter = None
        if self.ego_only_control_queries:
            self.ego_query_adapter = nn.Sequential(
                nn.LayerNorm(self.joint_token_dim),
                nn.Linear(self.joint_token_dim, self.joint_token_dim),
                nn.GELU(),
                nn.Linear(self.joint_token_dim, self.decoder_query_count * self.joint_token_dim),
            )
        self.joint_to_plane_xy = nn.Sequential(
            nn.LayerNorm(self.joint_token_dim),
            nn.Linear(self.joint_token_dim, self.token_dim),
        )
        self.joint_to_plane_xz = nn.Sequential(
            nn.LayerNorm(self.joint_token_dim),
            nn.Linear(self.joint_token_dim, self.token_dim),
        )
        self.joint_to_plane_yz = nn.Sequential(
            nn.LayerNorm(self.joint_token_dim),
            nn.Linear(self.joint_token_dim, self.token_dim),
        )
        self.fuser = GatedResidualDeformFusion(self.plane_channels)
        self.geo_convs = TriplaneGroupResnetBlock(
            self.plane_channels,
            out_channels=64,
            ks=5,
            input_norm=False,
            input_act=False,
        )
        self.semantic_decoder = MODELS.build(triplane_decoder_cfg) if triplane_decoder_cfg is not None else None
        self.pos_num_freq = 6
        self.learnable_control_queries.requires_grad = False
        for module in [self.learnable_control_adapter, self.control_query_norm]:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False

        self.history_tokens_xy = 1 + self.tokenizer_xy.num_tokens
        self.history_tokens_xz = 1 + self.tokenizer_xz.num_tokens
        self.history_tokens_yz = 1 + self.tokenizer_yz.num_tokens

    def _prepare_plane_sequence(self, plane):
        if plane.dim() == 5:
            return plane
        if plane.dim() == 4:
            return plane.unsqueeze(0)
        raise ValueError(f'unsupported plane shape: {tuple(plane.shape)}')

    def _build_control_features(self, metas, future_steps, device):
        if metas is None or len(metas) == 0:
            return torch.zeros((1, future_steps, self.control_dim), dtype=torch.float32, device=device)

        controls = []
        for meta in metas:
            rel_poses = torch.as_tensor(meta.get('rel_poses', None), dtype=torch.float32, device=device)
            if rel_poses.ndim != 2 or rel_poses.shape[0] < self.prev_steps + future_steps:
                rel_poses = torch.zeros((self.prev_steps + future_steps, 2), dtype=torch.float32, device=device)
            xy_delta = rel_poses[self.prev_steps:self.prev_steps + future_steps, :2]

            rot_seq = torch.as_tensor(meta.get('ego2global_rotation_seq', None), dtype=torch.float32, device=device)
            if rot_seq.ndim == 2 and rot_seq.shape[0] >= self.prev_steps + future_steps + 1 and rot_seq.shape[1] == 4:
                yaw_seq = quat_wxyz_to_yaw(rot_seq[:self.prev_steps + future_steps + 1])
                yaw_delta = wrap_angle(
                    yaw_seq[self.prev_steps + 1:self.prev_steps + future_steps + 1] -
                    yaw_seq[self.prev_steps:self.prev_steps + future_steps]
                ).unsqueeze(-1)
            else:
                yaw_delta = torch.zeros((future_steps, 1), dtype=torch.float32, device=device)

            ctrl = torch.cat([xy_delta, yaw_delta], dim=-1)
            if ctrl.shape[-1] != self.control_dim:
                raise ValueError(
                    f'control feature dim mismatch: expected {self.control_dim}, got {int(ctrl.shape[-1])}. '
                    'Set control_dim=3 for [dx, dy, yaw_rad].'
                )
            controls.append(ctrl)
        return torch.stack(controls, dim=0)

    def _encode_plane_tokens(self, tokenizer, plane_seq):
        batch_size, total_steps = plane_seq.shape[:2]
        flat = plane_seq.reshape(batch_size * total_steps, plane_seq.shape[2], plane_seq.shape[3], plane_seq.shape[4])
        tokens = tokenizer.encode(flat)
        return tokens.reshape(batch_size, total_steps, tokens.shape[1], tokens.shape[2])

    def _encode_triplanes_from_occs(self, input_occs):
        batch_size, total_steps, height, width, depth = input_occs.shape
        occs = input_occs.long()
        feat = self.triplane_class_embeds(occs)
        feat = feat.reshape(batch_size * total_steps, height, width, depth, self.plane_channels).permute(0, 4, 1, 2, 3)
        vol_feat = self.triplane_encoder(feat)

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)

        xy_feat = (self.triplane_encoder_norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.triplane_encoder_norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.triplane_encoder_norm(yz_feat) * 0.5).tanh()

        xy_feat = xy_feat.view(batch_size, total_steps, xy_feat.shape[1], xy_feat.shape[2], xy_feat.shape[3])
        xz_feat = xz_feat.view(batch_size, total_steps, xz_feat.shape[1], xz_feat.shape[2], xz_feat.shape[3])
        yz_feat = yz_feat.view(batch_size, total_steps, yz_feat.shape[1], yz_feat.shape[2], yz_feat.shape[3])
        return xy_feat, xz_feat, yz_feat

    def _encode_single_plane_tokens(self, tokenizer, plane):
        return tokenizer.encode(plane)

    def _build_control_token_sequence(self, control_feats):
        ego_control_tokens = self.control_to_joint(self.control_token_mlp(control_feats))
        batch_size, future_steps, _ = ego_control_tokens.shape
        if self.real_control_adapter is not None:
            real_control_tokens = self.real_control_adapter(ego_control_tokens.reshape(batch_size * future_steps, -1))
            real_control_tokens = real_control_tokens.view(
                batch_size, future_steps, self.num_real_control_tokens, self.joint_token_dim
            )
        else:
            real_control_tokens = ego_control_tokens.unsqueeze(2)
        external_control_tokens = self.external_control_norm(self.external_control_queries).unsqueeze(1).expand(
            batch_size, future_steps, -1, -1
        )
        return real_control_tokens, external_control_tokens

    def _build_ego_query_tokens(self, ego_token):
        if self.ego_query_adapter is None:
            raise RuntimeError('ego_query_adapter is not initialized')
        batch_size = int(ego_token.shape[0])
        query_tokens = self.ego_query_adapter(ego_token.squeeze(1))
        return query_tokens.view(batch_size, self.decoder_query_count, self.joint_token_dim)

    def _summarize_control_attention(self, step_tokens, control_token, ego_only_queries=False):
        logits = torch.matmul(
            F.normalize(step_tokens, dim=-1),
            F.normalize(control_token, dim=-1).transpose(-1, -2),
        ) / math.sqrt(max(1, self.joint_token_dim))
        attn = torch.softmax(logits, dim=-1)
        if ego_only_queries:
            return {
                'all': attn,
                'real': attn.mean(dim=(-2, -1)),
                'learnable': torch.zeros((attn.shape[0],), dtype=attn.dtype, device=attn.device),
            }
        return {
            'all': attn,
            'real': attn[..., :1].mean(dim=(-2, -1)),
            'learnable': attn[..., 1:].mean(dim=(-2, -1)) if attn.shape[-1] > 1 else torch.zeros(
                (attn.shape[0],), dtype=attn.dtype, device=attn.device
            ),
        }

    def _direct_control_attention(self, control_tokens, history_tokens):
        if history_tokens.shape[1] == 0:
            return history_tokens.new_zeros((history_tokens.shape[0], control_tokens.shape[1], 0))
        logits = torch.matmul(
            F.normalize(control_tokens, dim=-1),
            F.normalize(history_tokens, dim=-1).transpose(-1, -2),
        ) / math.sqrt(max(1, self.joint_token_dim))
        return torch.softmax(logits, dim=-1)

    def _control_regions_from_memory(self, control_tokens, memory, history_len, memory_attn=None):
        if memory_attn is None:
            logits = torch.matmul(
                F.normalize(control_tokens, dim=-1),
                F.normalize(memory, dim=-1).transpose(-1, -2),
            ) / math.sqrt(max(1, self.joint_token_dim))
            attn = torch.softmax(logits, dim=-1)
        else:
            attn = memory_attn

        xy_tokens = self.tokenizer_xy.num_tokens
        xz_tokens = self.tokenizer_xz.num_tokens
        yz_tokens = self.tokenizer_yz.num_tokens
        per_frame_tokens = xy_tokens + xz_tokens + yz_tokens
        attn = attn.reshape(attn.shape[0], attn.shape[1], int(history_len), per_frame_tokens)

        xy_region = attn[..., :xy_tokens].sum(dim=2)
        xz_region = attn[..., xy_tokens:xy_tokens + xz_tokens].sum(dim=2)
        yz_region = attn[..., xy_tokens + xz_tokens:].sum(dim=2)
        eps = torch.finfo(attn.dtype).eps
        xy_region = xy_region / xy_region.sum(dim=-1, keepdim=True).clamp_min(eps)
        xz_region = xz_region / xz_region.sum(dim=-1, keepdim=True).clamp_min(eps)
        yz_region = yz_region / yz_region.sum(dim=-1, keepdim=True).clamp_min(eps)
        return xy_region, xz_region, yz_region

    def _get_ego_pose_sequence(self, metas, total_steps, device, dtype):
        batch_rot = []
        batch_trans = []
        if metas is None:
            metas = []
        for meta in metas:
            trans = torch.as_tensor(meta.get('ego2global_translation_seq', None), dtype=dtype, device=device)
            rot = torch.as_tensor(meta.get('ego2global_rotation_seq', None), dtype=dtype, device=device)
            if trans.ndim != 2 or trans.shape[0] < total_steps:
                trans = torch.zeros((total_steps, 3), dtype=dtype, device=device)
            else:
                trans = trans[:total_steps]
            if rot.ndim != 2 or rot.shape[0] < total_steps:
                rot_mats = torch.eye(3, dtype=dtype, device=device).view(1, 3, 3).repeat(total_steps, 1, 1)
            else:
                rot_mats = quat_wxyz_to_rotmat(rot[:total_steps])
            batch_rot.append(rot_mats)
            batch_trans.append(trans)

        if len(batch_rot) == 0:
            rot_mats = torch.eye(3, dtype=dtype, device=device).view(1, 1, 3, 3).repeat(1, total_steps, 1, 1)
            trans = torch.zeros((1, total_steps, 3), dtype=dtype, device=device)
            return rot_mats, trans
        return torch.stack(batch_rot, dim=0), torch.stack(batch_trans, dim=0)

    def _plane_grid_to_metric(self, plane_hw, axes, device, dtype):
        h, w = int(plane_hw[0]), int(plane_hw[1])
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype),
            indexing='ij',
        )
        grid_norm = torch.stack([yy, xx], dim=-1)
        half_extent = torch.tensor([40.0, 40.0, 3.2], dtype=dtype, device=device)
        center = torch.tensor([0.0, 0.0, 2.2], dtype=dtype, device=device)
        coords = center.view(1, 1, 3).expand(h, w, 3).clone()
        coords[..., axes[0]] = grid_norm[..., 0] * half_extent[axes[0]] + center[axes[0]]
        coords[..., axes[1]] = grid_norm[..., 1] * half_extent[axes[1]] + center[axes[1]]
        return coords, half_extent, center

    def _warp_plane_between_ego_frames(self, plane, rot_src, trans_src, rot_tgt, trans_tgt, axes):
        batch_size, _, height, width = plane.shape
        coords_tgt, half_extent, center = self._plane_grid_to_metric(
            (height, width), axes, plane.device, plane.dtype
        )
        coords_tgt = coords_tgt.view(1, height * width, 3).expand(batch_size, -1, -1)
        coords_global = torch.bmm(coords_tgt, rot_tgt.transpose(1, 2)) + trans_tgt.unsqueeze(1)
        coords_src = torch.bmm(coords_global - trans_src.unsqueeze(1), rot_src)
        sample = torch.empty((batch_size, height * width, 2), dtype=plane.dtype, device=plane.device)
        sample[..., 0] = (coords_src[..., axes[1]] - center[axes[1]]) / half_extent[axes[1]]
        sample[..., 1] = (coords_src[..., axes[0]] - center[axes[0]]) / half_extent[axes[0]]
        grid = sample.view(batch_size, height, width, 2)
        return F.grid_sample(
            plane,
            grid,
            mode=self.warp_sample_mode,
            padding_mode=self.warp_padding_mode,
            align_corners=True,
        )

    def _sample_feature_plane2d(self, feat_map, coords):
        sample_coords = coords.view(coords.shape[0], 1, -1, 2)
        feat = F.grid_sample(
            feat_map,
            sample_coords.flip(-1),
            align_corners=False,
            padding_mode='border',
        )
        feat = feat[:, :, 0, :].transpose(1, 2)
        return feat

    def _triplane_query_feat(self, feat_maps, querys):
        coords_list = [[0, 1], [0, 2], [1, 2]]
        geo_feat_maps = [fm[:, :self.plane_channels] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        h_geo = 0
        for i in range(3):
            h_geo = h_geo + self._sample_feature_plane2d(
                geo_feat_maps[i], querys[..., coords_list[i]]
            )

        pe = []
        for freq in range(self.pos_num_freq):
            pe.append(torch.sin((2.0 ** freq) * querys))
            pe.append(torch.cos((2.0 ** freq) * querys))
        pe = torch.cat(pe, dim=-1)
        return torch.cat([h_geo, pe], dim=-1)

    def _decode_semantics(self, pred_xys, pred_xzs, pred_yzs, querys, xyz_centers):
        if self.semantic_decoder is None:
            return {}

        batch_size, future_steps = pred_xys.shape[:2]
        num_queries = int(querys.shape[2])

        flat_xy = pred_xys.reshape(batch_size * future_steps, pred_xys.shape[2], pred_xys.shape[3], pred_xys.shape[4])
        flat_xz = pred_xzs.reshape(batch_size * future_steps, pred_xzs.shape[2], pred_xzs.shape[3], pred_xzs.shape[4])
        flat_yz = pred_yzs.reshape(batch_size * future_steps, pred_yzs.shape[2], pred_yzs.shape[3], pred_yzs.shape[4])
        flat_querys = querys[:, self.prev_steps:].reshape(batch_size * future_steps, num_queries, querys.shape[-1])
        flat_centers = xyz_centers[:, self.prev_steps:].reshape(batch_size * future_steps, num_queries, xyz_centers.shape[-1]).long()

        qfeat = self._triplane_query_feat([flat_xy, flat_xz, flat_yz], flat_querys)
        preds = self.semantic_decoder(qfeat)
        softmax_preds = torch.softmax(preds, dim=-1)

        pred_logits = torch.zeros(
            (batch_size * future_steps, 200, 200, 16, self.num_cls),
            dtype=preds.dtype,
            device=preds.device,
        )
        pred_logits[:, :, :, :, -1] = 1.0
        pred_output = torch.zeros(
            (batch_size * future_steps, 200, 200, 16, self.num_cls),
            dtype=softmax_preds.dtype,
            device=softmax_preds.device,
        )
        pred_output[:, :, :, :, -1] = 1.0

        for i in range(batch_size * future_steps):
            centers = flat_centers[i]
            pred_output[
                i,
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                :,
            ] = softmax_preds[i]
            pred_logits[
                i,
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                :,
            ] = preds[i]

        pred_logits = pred_logits.view(batch_size, future_steps, 200, 200, 16, self.num_cls)
        pred_output = pred_output.view(batch_size, future_steps, 200, 200, 16, self.num_cls)
        sem_pred = pred_output.argmax(dim=-1)
        iou_pred = deepcopy(sem_pred)
        iou_pred[iou_pred != 17] = 1
        iou_pred[iou_pred == 17] = 0

        return {
            'preds': preds.view(batch_size, future_steps, num_queries, self.num_cls),
            'pred_logits': pred_logits,
            'pred_output': pred_output,
            'sem_pred': sem_pred,
            'iou_pred': iou_pred,
        }

    def load_pretrained_decoder(self, ckpt_path, logger=None):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        mapped = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                mapped['triplane_encoder.' + key[len('encoder.'):]] = value
            elif key.startswith('class_embeds.'):
                mapped['triplane_class_embeds.' + key[len('class_embeds.'):]] = value
            elif key.startswith('norm.'):
                mapped['triplane_encoder_norm.' + key[len('norm.'):]] = value
            if key.startswith('decoder.'):
                mapped['semantic_decoder.' + key[len('decoder.'):]] = value
            elif key.startswith('geo_convs.'):
                mapped['geo_convs.' + key[len('geo_convs.'):]] = value
            elif key.startswith('triplane_net.encoder.'):
                mapped['triplane_encoder.' + key[len('triplane_net.encoder.'):]] = value
            elif key.startswith('triplane_net.class_embeds.'):
                mapped['triplane_class_embeds.' + key[len('triplane_net.class_embeds.'):]] = value
            elif key.startswith('triplane_net.norm.'):
                mapped['triplane_encoder_norm.' + key[len('triplane_net.norm.'):]] = value
            elif key.startswith('triplane_net.decoder.'):
                mapped['semantic_decoder.' + key[len('triplane_net.decoder.'):]] = value
            elif key.startswith('triplane_net.geo_convs.'):
                mapped['geo_convs.' + key[len('triplane_net.geo_convs.'):]] = value

        msg = self.load_state_dict(mapped, strict=False)
        if logger is not None:
            logger.info(f'loaded pretrained decoder from: {ckpt_path}')
            logger.info(str(msg))
        return msg

    def forward(self, input_occs, querys, xys, xzs, yzs, xyz_labels=None, xyz_centers=None, metas=None):
        del xyz_labels, xys, xzs, yzs
        xys, xzs, yzs = self._encode_triplanes_from_occs(input_occs)

        batch_size, total_steps = xys.shape[:2]
        future_steps = max(0, total_steps - self.prev_steps)
        if future_steps <= 0:
            raise ValueError(f'need total_steps > prev_steps, got total_steps={total_steps}, prev_steps={self.prev_steps}')

        xy_hist = xys[:, :self.prev_steps]
        xz_hist = xzs[:, :self.prev_steps]
        yz_hist = yzs[:, :self.prev_steps]

        xy_hist_tokens = self._encode_plane_tokens(self.tokenizer_xy, xy_hist)
        xz_hist_tokens = self._encode_plane_tokens(self.tokenizer_xz, xz_hist)
        yz_hist_tokens = self._encode_plane_tokens(self.tokenizer_yz, yz_hist)
        hist_xy_tokens = [xy_hist_tokens[:, i] for i in range(self.prev_steps)]
        hist_xz_tokens = [xz_hist_tokens[:, i] for i in range(self.prev_steps)]
        hist_yz_tokens = [yz_hist_tokens[:, i] for i in range(self.prev_steps)]

        control_feats = self._build_control_features(metas, future_steps, xys.device)
        real_control_tokens, external_control_tokens = self._build_control_token_sequence(control_feats)

        out_shapes = {
            'xy': xys.shape[-2:],
            'xz': xzs.shape[-2:],
            'yz': yzs.shape[-2:],
        }

        latest_xy = xy_hist[:, self.prev_steps - 1]
        latest_xz = xz_hist[:, self.prev_steps - 1]
        latest_yz = yz_hist[:, self.prev_steps - 1]
        rot_mats = None
        trans = None
        if self.use_motion_compensation:
            rot_mats, trans = self._get_ego_pose_sequence(
                metas, total_steps, xys.device, xys.dtype
            )
            if rot_mats.shape[0] != batch_size:
                rot_mats = rot_mats[:1].expand(batch_size, -1, -1, -1)
                trans = trans[:1].expand(batch_size, -1, -1)

        pred_xys = []
        pred_xzs = []
        pred_yzs = []
        pred_delta_tokens_xy = []
        pred_delta_tokens_xz = []
        pred_delta_tokens_yz = []
        control_attn_all = []
        control_attn_real = []
        control_attn_learnable = []
        ego_xy_attn_all = []
        ego_xz_attn_all = []
        ego_yz_attn_all = []
        ext_xy_attn_all = []
        ext_xz_attn_all = []
        ext_yz_attn_all = []
        control_region_xy = []
        control_region_xz = []
        control_region_yz = []

        for step_idx in range(future_steps):
            ego_token = real_control_tokens[:, step_idx]
            ext_token = external_control_tokens[:, step_idx]

            recent_xy = hist_xy_tokens[-self.context_frames:]
            recent_xz = hist_xz_tokens[-self.context_frames:]
            recent_yz = hist_yz_tokens[-self.context_frames:]
            history_xy_seq = torch.stack(recent_xy, dim=1)
            history_xz_seq = torch.stack(recent_xz, dim=1)
            history_yz_seq = torch.stack(recent_yz, dim=1)

            history_xy = history_xy_seq.reshape(batch_size, -1, self.joint_token_dim)
            history_xz = history_xz_seq.reshape(batch_size, -1, self.joint_token_dim)
            history_yz = history_yz_seq.reshape(batch_size, -1, self.joint_token_dim)
            ego_xy_attn = self._direct_control_attention(ego_token, history_xy)
            ego_xz_attn = self._direct_control_attention(ego_token, history_xz)
            ego_yz_attn = self._direct_control_attention(ego_token, history_yz)
            ext_xy_attn = self._direct_control_attention(ext_token, history_xy)
            ext_xz_attn = self._direct_control_attention(ext_token, history_xz)
            ext_yz_attn = self._direct_control_attention(ext_token, history_yz)

            memory = self.triplane_transformer.encode_memory(
                history_xy_seq,
                history_xz_seq,
                history_yz_seq,
            )
            if self.control_query_only_decoder:
                if self.ego_only_control_queries:
                    query_tokens = self._build_ego_query_tokens(ego_token)
                    xy_joint_tokens, xz_joint_tokens, yz_joint_tokens, decoded_control_tokens, control_cross_attn = (
                        self.triplane_transformer.decode_query_tokens(memory, query_tokens, step_idx)
                    )
                else:
                    xy_joint_tokens, xz_joint_tokens, yz_joint_tokens, decoded_control_tokens, control_cross_attn = (
                        self.triplane_transformer.decode_control_delta(memory, ego_token, ext_token, step_idx)
                    )
            else:
                xy_joint_tokens, xz_joint_tokens, yz_joint_tokens, decoded_control_tokens, control_cross_attn = (
                    self.triplane_transformer.decode_delta(memory, ego_token, ext_token, step_idx)
                )
            region_xy, region_xz, region_yz = self._control_regions_from_memory(
                decoded_control_tokens,
                memory,
                history_xy_seq.shape[1],
                memory_attn=control_cross_attn,
            )

            xy_delta_tokens = self.joint_to_plane_xy(xy_joint_tokens)
            xz_delta_tokens = self.joint_to_plane_xz(xz_joint_tokens)
            yz_delta_tokens = self.joint_to_plane_yz(yz_joint_tokens)

            if self.use_motion_compensation:
                pose_src_idx = self.prev_steps - 1 + step_idx
                pose_tgt_idx = self.prev_steps + step_idx
                aligned_xy = self._warp_plane_between_ego_frames(
                    latest_xy,
                    rot_mats[:, pose_src_idx],
                    trans[:, pose_src_idx],
                    rot_mats[:, pose_tgt_idx],
                    trans[:, pose_tgt_idx],
                    (0, 1),
                )
                aligned_xz = self._warp_plane_between_ego_frames(
                    latest_xz,
                    rot_mats[:, pose_src_idx],
                    trans[:, pose_src_idx],
                    rot_mats[:, pose_tgt_idx],
                    trans[:, pose_tgt_idx],
                    (0, 2),
                )
                aligned_yz = self._warp_plane_between_ego_frames(
                    latest_yz,
                    rot_mats[:, pose_src_idx],
                    trans[:, pose_src_idx],
                    rot_mats[:, pose_tgt_idx],
                    trans[:, pose_tgt_idx],
                    (1, 2),
                )
            else:
                aligned_xy = latest_xy
                aligned_xz = latest_xz
                aligned_yz = latest_yz
            xy_pred = self.tokenizer_xy.decode(xy_delta_tokens, out_shapes['xy'])
            xz_pred = self.tokenizer_xz.decode(xz_delta_tokens, out_shapes['xz'])
            yz_pred = self.tokenizer_yz.decode(yz_delta_tokens, out_shapes['yz'])

            if self.predict_delta:
                next_xy_raw = aligned_xy + xy_pred
                next_xz_raw = aligned_xz + xz_pred
                next_yz_raw = aligned_yz + yz_pred
            else:
                next_xy_raw = xy_pred
                next_xz_raw = xz_pred
                next_yz_raw = yz_pred
            next_xy, next_xz, next_yz = self.fuser(next_xy_raw, next_xz_raw, next_yz_raw)

            latest_xy_tokens = self._encode_single_plane_tokens(self.tokenizer_xy, next_xy)
            latest_xz_tokens = self._encode_single_plane_tokens(self.tokenizer_xz, next_xz)
            latest_yz_tokens = self._encode_single_plane_tokens(self.tokenizer_yz, next_yz)
            latest_xy = next_xy
            latest_xz = next_xz
            latest_yz = next_yz

            hist_xy_tokens.append(latest_xy_tokens)
            hist_xz_tokens.append(latest_xz_tokens)
            hist_yz_tokens.append(latest_yz_tokens)

            joint_delta = decoded_latent_tokens
            joint_control = decoded_control_tokens
            attn_stats = self._summarize_control_attention(
                joint_delta,
                joint_control,
                ego_only_queries=self.ego_only_control_queries and self.control_query_only_decoder,
            )

            pred_xys.append(next_xy)
            pred_xzs.append(next_xz)
            pred_yzs.append(next_yz)
            pred_delta_tokens_xy.append(xy_delta_tokens)
            pred_delta_tokens_xz.append(xz_delta_tokens)
            pred_delta_tokens_yz.append(yz_delta_tokens)
            control_attn_all.append(attn_stats['all'])
            control_attn_real.append(attn_stats['real'])
            control_attn_learnable.append(attn_stats['learnable'])
            ego_xy_attn_all.append(ego_xy_attn)
            ego_xz_attn_all.append(ego_xz_attn)
            ego_yz_attn_all.append(ego_yz_attn)
            ext_xy_attn_all.append(ext_xy_attn)
            ext_xz_attn_all.append(ext_xz_attn)
            ext_yz_attn_all.append(ext_yz_attn)
            control_region_xy.append(region_xy)
            control_region_xz.append(region_xz)
            control_region_yz.append(region_yz)

        pred_xys = torch.stack(pred_xys, dim=1)
        pred_xzs = torch.stack(pred_xzs, dim=1)
        pred_yzs = torch.stack(pred_yzs, dim=1)
        pred_delta_tokens_xy = torch.stack(pred_delta_tokens_xy, dim=1)
        pred_delta_tokens_xz = torch.stack(pred_delta_tokens_xz, dim=1)
        pred_delta_tokens_yz = torch.stack(pred_delta_tokens_yz, dim=1)
        control_attn_all = torch.stack(control_attn_all, dim=1)
        control_attn_real = torch.stack(control_attn_real, dim=1)
        control_attn_learnable = torch.stack(control_attn_learnable, dim=1)
        control_region_xy = torch.stack(control_region_xy, dim=1)
        control_region_xz = torch.stack(control_region_xz, dim=1)
        control_region_yz = torch.stack(control_region_yz, dim=1)
        if self.ego_only_control_queries and self.control_query_only_decoder:
            learnable_control_tokens_out = real_control_tokens.new_zeros(
                real_control_tokens.shape[0],
                real_control_tokens.shape[1],
                0,
                real_control_tokens.shape[-1],
            )
        else:
            learnable_control_tokens_out = external_control_tokens

        output = {
            'hexplane': [pred_xys.squeeze(0), pred_xzs.squeeze(0), pred_yzs.squeeze(0)],
            'delta_tokens_xy': pred_delta_tokens_xy,
            'delta_tokens_xz': pred_delta_tokens_xz,
            'delta_tokens_yz': pred_delta_tokens_yz,
            'ego_control_tokens': real_control_tokens,
            'external_control_tokens': external_control_tokens,
            'real_control_tokens': real_control_tokens,
            'learnable_control_tokens': learnable_control_tokens_out,
            'control_token_attn': control_attn_all,
            'control_token_attn_real': control_attn_real,
            'control_token_attn_learnable': control_attn_learnable,
            'ego_control_attn_xy': ego_xy_attn_all,
            'ego_control_attn_xz': ego_xz_attn_all,
            'ego_control_attn_yz': ego_yz_attn_all,
            'external_control_attn_xy': ext_xy_attn_all,
            'external_control_attn_xz': ext_xz_attn_all,
            'external_control_attn_yz': ext_yz_attn_all,
            'control_region_xy': control_region_xy,
            'control_region_xz': control_region_xz,
            'control_region_yz': control_region_yz,
        }
        if querys is not None and xyz_centers is not None:
            output.update(self._decode_semantics(pred_xys, pred_xzs, pred_yzs, querys, xyz_centers))
        return output


@MODELS.register_module()
class DeltaTokenPredictorV2(DeltaTokenPredictor):
    def __init__(self, *args, **kwargs):
        kwargs['control_query_only_decoder'] = True
        super().__init__(*args, **kwargs)


@MODELS.register_module()
class DeltaTokenPredictorV21(DeltaTokenPredictorV2):
    def __init__(self, *args, encoder_grid_size=(12, 12, 4), **kwargs):
        grid_size = tuple(int(v) for v in encoder_grid_size)
        kwargs['decoder_query_count'] = int(grid_size[0] * grid_size[1] * grid_size[2])
        super().__init__(*args, **kwargs)
        self.encoder_grid_size = grid_size
        self.point_tokenizer_3d = TriplanePointTokenizer3D(
            in_channels=self.plane_channels,
            token_dim=self.joint_token_dim,
            grid_size=self.encoder_grid_size,
        )
        self.history_tokens_xyz = self.point_tokenizer_3d.num_tokens
        self.latent_to_triplane = LowRes3DToTriplaneProjector(
            token_dim=self.joint_token_dim,
            plane_channels=self.plane_channels,
        )

    def _encode_triplane_point_tokens(self, xy_seq, xz_seq, yz_seq):
        batch_size, total_steps = xy_seq.shape[:2]
        flat_xy = xy_seq.reshape(batch_size * total_steps, xy_seq.shape[2], xy_seq.shape[3], xy_seq.shape[4])
        flat_xz = xz_seq.reshape(batch_size * total_steps, xz_seq.shape[2], xz_seq.shape[3], xz_seq.shape[4])
        flat_yz = yz_seq.reshape(batch_size * total_steps, yz_seq.shape[2], yz_seq.shape[3], yz_seq.shape[4])
        tokens, coords = self.point_tokenizer_3d.encode(flat_xy, flat_xz, flat_yz)
        tokens = tokens.reshape(batch_size, total_steps, tokens.shape[1], tokens.shape[2])
        return tokens, coords

    def _control_regions_from_point_memory(self, control_tokens, memory, history_len, memory_attn=None):
        if memory_attn is None:
            logits = torch.matmul(
                F.normalize(control_tokens, dim=-1),
                F.normalize(memory, dim=-1).transpose(-1, -2),
            ) / math.sqrt(max(1, self.joint_token_dim))
            attn = torch.softmax(logits, dim=-1)
        else:
            attn = memory_attn

        attn = attn.reshape(attn.shape[0], attn.shape[1], int(history_len), self.history_tokens_xyz)
        xyz_region = attn.sum(dim=2)
        eps = torch.finfo(attn.dtype).eps
        xyz_region = xyz_region / xyz_region.sum(dim=-1, keepdim=True).clamp_min(eps)
        return xyz_region

    def _warp_plane_between_ego_frames_with_mask(self, plane, rot_src, trans_src, rot_tgt, trans_tgt, axes):
        batch_size, _, height, width = plane.shape
        coords_tgt, half_extent, center = self._plane_grid_to_metric(
            (height, width), axes, plane.device, plane.dtype
        )
        coords_tgt = coords_tgt.view(1, height * width, 3).expand(batch_size, -1, -1)
        coords_global = torch.bmm(coords_tgt, rot_tgt.transpose(1, 2)) + trans_tgt.unsqueeze(1)
        coords_src = torch.bmm(coords_global - trans_src.unsqueeze(1), rot_src)
        sample = torch.empty((batch_size, height * width, 2), dtype=plane.dtype, device=plane.device)
        sample[..., 0] = (coords_src[..., axes[1]] - center[axes[1]]) / half_extent[axes[1]]
        sample[..., 1] = (coords_src[..., axes[0]] - center[axes[0]]) / half_extent[axes[0]]
        grid = sample.view(batch_size, height, width, 2)
        warped = F.grid_sample(
            plane,
            grid,
            mode=self.warp_sample_mode,
            padding_mode=self.warp_padding_mode,
            align_corners=True,
        )
        valid = (
            (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) &
            (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
        ).to(dtype=plane.dtype).unsqueeze(1)
        return warped, valid

    def _align_history_triplanes_to_target(
        self,
        recent_xy,
        recent_xz,
        recent_yz,
        recent_pose_indices,
        target_pose_idx,
        rot_mats,
        trans,
    ):
        aligned_xy = []
        aligned_xz = []
        aligned_yz = []
        for src_xy, src_xz, src_yz, src_pose_idx in zip(
            recent_xy, recent_xz, recent_yz, recent_pose_indices
        ):
            if int(src_pose_idx) == int(target_pose_idx):
                aligned_xy.append(src_xy)
                aligned_xz.append(src_xz)
                aligned_yz.append(src_yz)
                continue

            aligned_xy.append(
                self._warp_plane_between_ego_frames(
                    src_xy,
                    rot_mats[:, int(src_pose_idx)],
                    trans[:, int(src_pose_idx)],
                    rot_mats[:, int(target_pose_idx)],
                    trans[:, int(target_pose_idx)],
                    (0, 1),
                )
            )
            aligned_xz.append(
                self._warp_plane_between_ego_frames(
                    src_xz,
                    rot_mats[:, int(src_pose_idx)],
                    trans[:, int(src_pose_idx)],
                    rot_mats[:, int(target_pose_idx)],
                    trans[:, int(target_pose_idx)],
                    (0, 2),
                )
            )
            aligned_yz.append(
                self._warp_plane_between_ego_frames(
                    src_yz,
                    rot_mats[:, int(src_pose_idx)],
                    trans[:, int(src_pose_idx)],
                    rot_mats[:, int(target_pose_idx)],
                    trans[:, int(target_pose_idx)],
                    (1, 2),
                )
            )
        return aligned_xy, aligned_xz, aligned_yz

    def forward(self, input_occs, querys, xys, xzs, yzs, xyz_labels=None, xyz_centers=None, metas=None):
        del xyz_labels, xys, xzs, yzs
        xys, xzs, yzs = self._encode_triplanes_from_occs(input_occs)

        batch_size, total_steps = xys.shape[:2]
        future_steps = max(0, total_steps - self.prev_steps)
        if future_steps <= 0:
            raise ValueError(f'need total_steps > prev_steps, got total_steps={total_steps}, prev_steps={self.prev_steps}')

        xy_hist = xys[:, :self.prev_steps]
        xz_hist = xzs[:, :self.prev_steps]
        yz_hist = yzs[:, :self.prev_steps]

        _, point_token_coords = self._encode_triplane_point_tokens(xy_hist, xz_hist, yz_hist)

        hist_xy_planes = [xy_hist[:, i] for i in range(self.prev_steps)]
        hist_xz_planes = [xz_hist[:, i] for i in range(self.prev_steps)]
        hist_yz_planes = [yz_hist[:, i] for i in range(self.prev_steps)]
        hist_pose_indices = list(range(self.prev_steps))

        control_feats = self._build_control_features(metas, future_steps, xys.device)
        real_control_tokens, external_control_tokens = self._build_control_token_sequence(control_feats)

        out_shapes = {
            'xy': xys.shape[-2:],
            'xz': xzs.shape[-2:],
            'yz': yzs.shape[-2:],
        }

        latest_xy = xy_hist[:, self.prev_steps - 1]
        latest_xz = xz_hist[:, self.prev_steps - 1]
        latest_yz = yz_hist[:, self.prev_steps - 1]
        rot_mats = None
        trans = None
        if self.use_motion_compensation:
            rot_mats, trans = self._get_ego_pose_sequence(metas, total_steps, xys.device, xys.dtype)
            if rot_mats.shape[0] != batch_size:
                rot_mats = rot_mats[:1].expand(batch_size, -1, -1, -1)
                trans = trans[:1].expand(batch_size, -1, -1)

        pred_xys = []
        pred_xzs = []
        pred_yzs = []
        pred_delta_tokens_xy = []
        pred_delta_tokens_xz = []
        pred_delta_tokens_yz = []
        control_attn_all = []
        control_attn_real = []
        control_attn_learnable = []
        control_region_xyz = []
        pred_latent_tokens = []
        known_masks_xy = []
        known_masks_xz = []
        known_masks_yz = []

        for step_idx in range(future_steps):
            ego_token = real_control_tokens[:, step_idx]
            ext_token = external_control_tokens[:, step_idx]
            target_pose_idx = self.prev_steps + step_idx
            recent_xy = hist_xy_planes[-self.context_frames:]
            recent_xz = hist_xz_planes[-self.context_frames:]
            recent_yz = hist_yz_planes[-self.context_frames:]
            recent_pose_indices = hist_pose_indices[-self.context_frames:]

            if self.use_motion_compensation:
                aligned_hist_xy, aligned_hist_xz, aligned_hist_yz = self._align_history_triplanes_to_target(
                    recent_xy,
                    recent_xz,
                    recent_yz,
                    recent_pose_indices,
                    target_pose_idx,
                    rot_mats,
                    trans,
                )
            else:
                aligned_hist_xy, aligned_hist_xz, aligned_hist_yz = recent_xy, recent_xz, recent_yz

            history_point_seq, _ = self._encode_triplane_point_tokens(
                torch.stack(aligned_hist_xy, dim=1),
                torch.stack(aligned_hist_xz, dim=1),
                torch.stack(aligned_hist_yz, dim=1),
            )
            memory = self.triplane_transformer.encode_memory_tokens(history_point_seq)

            if self.ego_only_control_queries:
                query_tokens = self._build_ego_query_tokens(ego_token)
            else:
                query_tokens = torch.cat([ego_token, ext_token], dim=1)
            decoded_latent_tokens, control_cross_attn = self.triplane_transformer.decode_latent_queries(
                memory,
                query_tokens,
                step_idx,
            )
            decoded_control_tokens = decoded_latent_tokens

            region_xyz = self._control_regions_from_point_memory(
                decoded_control_tokens,
                memory,
                history_point_seq.shape[1],
                memory_attn=control_cross_attn,
            )
            xy_pred, xz_pred, yz_pred = self.latent_to_triplane(
                decoded_latent_tokens,
                self.encoder_grid_size,
                out_shapes,
            )
            xy_delta_tokens = self._encode_single_plane_tokens(self.tokenizer_xy, xy_pred)
            xz_delta_tokens = self._encode_single_plane_tokens(self.tokenizer_xz, xz_pred)
            yz_delta_tokens = self._encode_single_plane_tokens(self.tokenizer_yz, yz_pred)

            if self.use_motion_compensation:
                pose_src_idx = self.prev_steps - 1 + step_idx
                aligned_xy, known_xy = self._warp_plane_between_ego_frames_with_mask(
                    latest_xy,
                    rot_mats[:, pose_src_idx],
                    trans[:, pose_src_idx],
                    rot_mats[:, target_pose_idx],
                    trans[:, target_pose_idx],
                    (0, 1),
                )
                aligned_xz, known_xz = self._warp_plane_between_ego_frames_with_mask(
                    latest_xz,
                    rot_mats[:, pose_src_idx],
                    trans[:, pose_src_idx],
                    rot_mats[:, target_pose_idx],
                    trans[:, target_pose_idx],
                    (0, 2),
                )
                aligned_yz, known_yz = self._warp_plane_between_ego_frames_with_mask(
                    latest_yz,
                    rot_mats[:, pose_src_idx],
                    trans[:, pose_src_idx],
                    rot_mats[:, target_pose_idx],
                    trans[:, target_pose_idx],
                    (1, 2),
                )
            else:
                aligned_xy = latest_xy
                aligned_xz = latest_xz
                aligned_yz = latest_yz
                known_xy = torch.ones_like(latest_xy[:, :1])
                known_xz = torch.ones_like(latest_xz[:, :1])
                known_yz = torch.ones_like(latest_yz[:, :1])

            if self.predict_delta:
                next_xy_raw = aligned_xy + xy_pred
                next_xz_raw = aligned_xz + xz_pred
                next_yz_raw = aligned_yz + yz_pred
            else:
                next_xy_raw = xy_pred
                next_xz_raw = xz_pred
                next_yz_raw = yz_pred
            next_xy, next_xz, next_yz = self.fuser(next_xy_raw, next_xz_raw, next_yz_raw)

            latest_xy = next_xy
            latest_xz = next_xz
            latest_yz = next_yz

            hist_xy_planes.append(latest_xy)
            hist_xz_planes.append(latest_xz)
            hist_yz_planes.append(latest_yz)
            hist_pose_indices.append(target_pose_idx)

            joint_delta = decoded_latent_tokens
            joint_control = decoded_control_tokens
            attn_stats = self._summarize_control_attention(
                joint_delta,
                joint_control,
                ego_only_queries=self.ego_only_control_queries and self.control_query_only_decoder,
            )

            pred_xys.append(next_xy)
            pred_xzs.append(next_xz)
            pred_yzs.append(next_yz)
            pred_delta_tokens_xy.append(xy_delta_tokens)
            pred_delta_tokens_xz.append(xz_delta_tokens)
            pred_delta_tokens_yz.append(yz_delta_tokens)
            control_attn_all.append(attn_stats['all'])
            control_attn_real.append(attn_stats['real'])
            control_attn_learnable.append(attn_stats['learnable'])
            control_region_xyz.append(region_xyz)
            pred_latent_tokens.append(decoded_latent_tokens)
            known_masks_xy.append(known_xy)
            known_masks_xz.append(known_xz)
            known_masks_yz.append(known_yz)

        pred_xys = torch.stack(pred_xys, dim=1)
        pred_xzs = torch.stack(pred_xzs, dim=1)
        pred_yzs = torch.stack(pred_yzs, dim=1)
        pred_delta_tokens_xy = torch.stack(pred_delta_tokens_xy, dim=1)
        pred_delta_tokens_xz = torch.stack(pred_delta_tokens_xz, dim=1)
        pred_delta_tokens_yz = torch.stack(pred_delta_tokens_yz, dim=1)
        pred_latent_tokens = torch.stack(pred_latent_tokens, dim=1)
        control_attn_all = torch.stack(control_attn_all, dim=1)
        control_attn_real = torch.stack(control_attn_real, dim=1)
        control_attn_learnable = torch.stack(control_attn_learnable, dim=1)
        control_region_xyz = torch.stack(control_region_xyz, dim=1)
        known_masks_xy = torch.stack(known_masks_xy, dim=1)
        known_masks_xz = torch.stack(known_masks_xz, dim=1)
        known_masks_yz = torch.stack(known_masks_yz, dim=1)

        if self.ego_only_control_queries and self.control_query_only_decoder:
            learnable_control_tokens_out = real_control_tokens.new_zeros(
                real_control_tokens.shape[0],
                real_control_tokens.shape[1],
                0,
                real_control_tokens.shape[-1],
            )
        else:
            learnable_control_tokens_out = external_control_tokens

        output = {
            'hexplane': [pred_xys.squeeze(0), pred_xzs.squeeze(0), pred_yzs.squeeze(0)],
            'delta_tokens_xy': pred_delta_tokens_xy,
            'delta_tokens_xz': pred_delta_tokens_xz,
            'delta_tokens_yz': pred_delta_tokens_yz,
            'latent_tokens_3d': pred_latent_tokens,
            'ego_control_tokens': real_control_tokens,
            'external_control_tokens': external_control_tokens,
            'real_control_tokens': real_control_tokens,
            'learnable_control_tokens': learnable_control_tokens_out,
            'control_token_attn': control_attn_all,
            'control_token_attn_real': control_attn_real,
            'control_token_attn_learnable': control_attn_learnable,
            'control_region_xyz': control_region_xyz,
            'control_region_xyz_coords': point_token_coords,
            'encoder_token_grid_size': torch.tensor(self.encoder_grid_size, device=pred_xys.device),
            'known_mask_xy': known_masks_xy,
            'known_mask_xz': known_masks_xz,
            'known_mask_yz': known_masks_yz,
        }
        if querys is not None and xyz_centers is not None:
            output.update(self._decode_semantics(pred_xys, pred_xzs, pred_yzs, querys, xyz_centers))
        return output
