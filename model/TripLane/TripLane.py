import torch
import torch.nn as nn
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn.functional as F
from copy import deepcopy


import sys, os, pdb

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
            vqvae_cfg=None,
            init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.norm = nn.InstanceNorm2d(expansion)

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)

        self.geo_convs = TriplaneGroupResnetBlock(
            expansion, out_channels=64,
            ks=5, input_norm=False, input_act=False)

        self.class_embeds = nn.Embedding(num_classes, expansion)
        self.pos_num_freq = 6

        self.only_triplane = only_triplane

    def forward_encoder(self, xxx):
        bs, F, H, W, D = xxx.shape

        xxx = self.class_embeds(xxx)

        xxx = xxx.reshape(bs*F, H, W, D, self.expansion).permute(0, 4, 1, 2, 3)

        vol_feat = self.encoder(xxx)

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)


        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]

    def sample_feature_plane2D(self, feat_map, x):
        sample_coords = x.view(x.shape[0], 1, -1, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1), align_corners=False, padding_mode='border')
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
        feat_maps = self.forward_encoder(x)

        if self.only_triplane:
            return feat_maps

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
    def __init__(self, geo_feat_channels=8, z_down=True, all_down=False, padding_mode="replicate",
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
        elif self.z_down :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else :
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
