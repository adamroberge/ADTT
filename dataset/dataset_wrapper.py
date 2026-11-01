import numpy as np, torch
from torch.utils import data
import torch.nn.functional as F
from copy import deepcopy
from mmengine import MMLogger
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor

logger = MMLogger.get_instance('genocc')
from . import OPENOCC_DATAWRAPPER

import os, sys, pdb
class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

@OPENOCC_DATAWRAPPER.register_module()
class lidarqueryhex_dataset_nuscenes(data.Dataset):
    def __init__(
            self,
            in_dataset,
            phase='train',
        ):
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)

    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        input, target, triplanes, querys, xyz_labels, xyz_centers, metas = self.point_cloud_dataset[index]
        input = torch.from_numpy(input.copy())
        target = torch.from_numpy(target.copy())
        xys, xzs, yzs = [], [], []

        for one_frame in triplanes:
            if isinstance(one_frame, dict):
                xy = one_frame.get('xy')
                xz = one_frame.get('xz')
                yz = one_frame.get('yz')

                if xy is None and 'xy_static' in one_frame and 'xy_dynamic' in one_frame:
                    xy = torch.cat([one_frame['xy_static'], one_frame['xy_dynamic']], dim=0)
                if xz is None and 'xz_static' in one_frame and 'xz_dynamic' in one_frame:
                    xz = torch.cat([one_frame['xz_static'], one_frame['xz_dynamic']], dim=0)
                if yz is None and 'yz_static' in one_frame and 'yz_dynamic' in one_frame:
                    yz = torch.cat([one_frame['yz_static'], one_frame['yz_dynamic']], dim=0)
            elif isinstance(one_frame, (list, tuple)) and len(one_frame) == 3:
                xy, xz, yz = one_frame
            else:
                raise TypeError(f'Unsupported triplane frame type: {type(one_frame)}')

            if xy is None or xz is None or yz is None:
                raise KeyError('Missing xy/xz/yz triplane data in dataset sample')

            xys.append(torch.as_tensor(xy).detach().squeeze(0).float())
            xzs.append(torch.as_tensor(xz).detach().squeeze(0).float())
            yzs.append(torch.as_tensor(yz).detach().squeeze(0).float())

        xys = torch.stack(xys, dim=0).detach()
        xzs = torch.stack(xzs, dim=0).detach()
        yzs = torch.stack(yzs, dim=0).detach()

        if isinstance(querys, (list, tuple)):
            querys = torch.stack(querys)
        if isinstance(xyz_labels, (list, tuple)):
            xyz_labels = torch.stack(xyz_labels)
        if isinstance(xyz_centers, (list, tuple)):
            xyz_centers = torch.stack(xyz_centers)

        querys = querys.detach() if isinstance(querys, torch.Tensor) else querys
        xyz_labels = xyz_labels.detach() if isinstance(xyz_labels, torch.Tensor) else xyz_labels
        xyz_centers = xyz_centers.detach() if isinstance(xyz_centers, torch.Tensor) else xyz_centers

        return input, target, xys, xzs, yzs, querys, xyz_labels, xyz_centers, metas


@OPENOCC_DATAWRAPPER.register_module()
class lidarimgquery_dataset_nuscenes(data.Dataset):
    def __init__(
            self,
            in_dataset,
            phase='train',
        ):
        self.point_cloud_dataset = in_dataset
        self.phase = phase

        self.transform = transforms.ToTensor()
        model_name = 'facebook/dinov2-base'
        self.dino_processor = AutoImageProcessor.from_pretrained(model_name)

    def __len__(self):
        return len(self.point_cloud_dataset)

    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        input, target, querys, xyz_labels, xyz_centers, metas = self.point_cloud_dataset[index]
        input = torch.from_numpy(input.copy())
        target = torch.from_numpy(target.copy())
        querys = torch.stack(querys)
        xyz_labels = torch.stack(xyz_labels)
        xyz_centers = torch.stack(xyz_centers)

        new_metas = {}

        bs = querys.shape[0]
        cam_positions, cam_rotations = [], []
        for iii in range(bs):
            cam_positions.append(metas[iii]['cam_positions'].copy())
            cam_rotations.append(metas[iii]['cam_rotations'].copy())

        cam_positions = torch.from_numpy(np.array(cam_positions))
        cam_positions[:, :, 0] = cam_positions[:, :, 0] / 10
        cam_positions[:, :, 1] = cam_positions[:, :, 1] / 5
        cam_positions[:, :, 2] = cam_positions[:, :, 2] / 2


        cam_rotations = torch.from_numpy(np.array(cam_rotations))
        cam_rotations[:, :, 0] = cam_rotations[:, :, 0] / 180
        cam_rotations[:, :, 1] = cam_rotations[:, :, 1] / 90
        cam_rotations[:, :, 2] = cam_rotations[:, :, 2] / 180

        new_metas.update({
            'cam_positions': cam_positions,
            'cam_rotations': cam_rotations,
        })

        imgs = []
        seq_check = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
        for iii in range(bs):
            for one_cam_idx, one_cam_file in enumerate(metas[iii]['img_filename']):
                assert seq_check[one_cam_idx] in one_cam_file
                image = Image.open(one_cam_file)
                imgs.append(image)


        new_metas.update({
            'imgs': imgs
        })
        for iii in range(bs):
            new_metas.update({
                iii: metas[iii]
            })
        new_metas["scene_name"] = metas["scene_name"]
        return input, target, querys, xyz_labels, xyz_centers, new_metas


@OPENOCC_DATAWRAPPER.register_module()
class lidarquery_dataset_nuscenes(data.Dataset):
    def __init__(
            self,
            in_dataset,
            phase='train',
        ):
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)

    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        input, target, querys, xyz_labels, xyz_centers, metas = self.point_cloud_dataset[index]
        input = torch.from_numpy(input.copy())
        target = torch.from_numpy(target.copy())
        querys = torch.stack(querys)
        xyz_labels = torch.stack(xyz_labels)
        xyz_centers = torch.stack(xyz_centers)
        return input, target, querys, xyz_labels, xyz_centers, metas


@OPENOCC_DATAWRAPPER.register_module()
class tpvformer_dataset_nuscenes(data.Dataset):
    def __init__(
            self,
            in_dataset,
            phase='train',
        ):
        self.point_cloud_dataset = in_dataset
        self.phase = phase

    def __len__(self):
        return len(self.point_cloud_dataset)

    def to_tensor(self, imgs):
        imgs = np.stack(imgs).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs

    def __getitem__(self, index):
        input, target, metas = self.point_cloud_dataset[index]
        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        return input, target, metas


def custom_collate_fn_temporal(data):
    data_tuple = []
    for i, item in enumerate(data[0]):
        if isinstance(item, torch.Tensor):
            data_tuple.append(torch.stack([d[i] for d in data]))
        elif isinstance(item, (dict, str)):
            data_tuple.append([d[i] for d in data])
        elif item is None:
            data_tuple.append(None)
        else:
            raise NotImplementedError
    return data_tuple
