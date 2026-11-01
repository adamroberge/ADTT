import torch.nn as nn
from mmengine.registry import MODELS
from mmengine.model import BaseModule


import sys, os, pdb

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


@MODELS.register_module()
class PoseDecoder(BaseModule):

    def __init__(
            self,
            in_channels,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
            init_cfg = None):
        super().__init__(init_cfg)

        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1

        pose_decoder = []
        for _ in range(num_layers - 1):
            pose_decoder.extend([
                nn.Linear(in_channels, in_channels),
                nn.ReLU(True)])
        pose_decoder.append(nn.Linear(in_channels, num_modes*num_fut_ts*2))
        self.pose_decoder = nn.Sequential(*pose_decoder)

    def forward(self, x):
        rel_pose = self.pose_decoder(x).reshape(*x.shape[:-1], self.num_modes, 2)
        return rel_pose
