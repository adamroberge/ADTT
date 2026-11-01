import os, numpy as np, pickle
from pyquaternion import Quaternion
from copy import deepcopy
from . import OPENOCC_DATASET
import torch
from scipy.ndimage import distance_transform_edt
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, Box3DMode
import shutil
from scipy.spatial.transform import Rotation as R

import sys, os, pdb

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def compute_tdf(
    voxel_label: np.ndarray,
    trunc_distance: float = 3,
    trunc_value: float = -1,
    empty_label: int = 17,
) -> np.ndarray:
    free = voxel_label == int(empty_label)
    tdf = distance_transform_edt(free)

    tdf[tdf > trunc_distance] = trunc_value
    return tdf


def get_query(
    voxel_label,
    num_class=18,
    grid_size=(200, 200, 16),
    max_points=200000,
    empty_label: int = 17,
    boundary_distance: int = 2,
    occupied_ratio: float = 0.9,
):
    labels = torch.from_numpy(np.asarray(voxel_label)).long()
    empty_label = int(empty_label)
    num_class = int(num_class)

    coords_occ = torch.nonzero(labels != empty_label, as_tuple=False)
    labs_occ = labels[coords_occ[:, 0], coords_occ[:, 1], coords_occ[:, 2]] if coords_occ.numel() > 0 else None

    quota_occ_total = int(max_points * float(occupied_ratio))
    quota_occ_per_class = max(1, quota_occ_total // max(1, (num_class - 1)))
    quota_empty = max_points - quota_occ_total

    xyzl = []
    for c in range(num_class):
        if c == empty_label:
            continue
        if coords_occ.numel() == 0:
            continue
        cls_coords = coords_occ[labs_occ == c]
        if cls_coords.numel() == 0:
            continue
        n = min(int(cls_coords.shape[0]), int(quota_occ_per_class))
        if n <= 0:
            continue
        idx = torch.randperm(int(cls_coords.shape[0]))[:n]
        picked = cls_coords[idx]
        xyzl.append(torch.nn.functional.pad(picked, (1, 0), 'constant', value=int(c)))

    tdf = compute_tdf(voxel_label, trunc_distance=float(boundary_distance), trunc_value=-1, empty_label=empty_label)
    near_mask = np.logical_and(tdf > 0, tdf <= float(boundary_distance))
    far_mask = np.logical_and(np.asarray(voxel_label) == empty_label, tdf == -1)
    near_coords = torch.nonzero(torch.from_numpy(near_mask), as_tuple=False)
    far_coords = torch.nonzero(torch.from_numpy(far_mask), as_tuple=False)

    empty_need = int(quota_empty)
    if empty_need > 0:
        near_take = min(int(near_coords.shape[0]), int(empty_need * 0.6)) if near_coords.numel() > 0 else 0
        far_take = min(int(far_coords.shape[0]), int(empty_need - near_take)) if far_coords.numel() > 0 else 0
        if near_take > 0:
            idx = torch.randperm(int(near_coords.shape[0]))[:near_take]
            xyzl.append(torch.nn.functional.pad(near_coords[idx], (1, 0), 'constant', value=empty_label))
        if far_take > 0:
            idx = torch.randperm(int(far_coords.shape[0]))[:far_take]
            xyzl.append(torch.nn.functional.pad(far_coords[idx], (1, 0), 'constant', value=empty_label))

    if len(xyzl) == 0:
        all_coords = torch.nonzero(torch.ones_like(labels, dtype=torch.bool), as_tuple=False)
        idx = torch.randperm(int(all_coords.shape[0]))[:max_points]
        picked = all_coords[idx]
        xyzl = torch.nn.functional.pad(picked, (1, 0), 'constant', value=empty_label)
    else:
        xyzl = torch.cat(xyzl, dim=0)
        if xyzl.shape[0] < max_points:
            remain = int(max_points - xyzl.shape[0])
            pool = far_coords if far_coords.numel() > 0 else (near_coords if near_coords.numel() > 0 else coords_occ)
            if pool is not None and pool.numel() > 0 and remain > 0:
                idx = torch.randint(0, int(pool.shape[0]), (remain,), dtype=torch.long)
                fill = pool[idx]
                if pool is coords_occ:
                    fill_labels = labels[fill[:, 0], fill[:, 1], fill[:, 2]].long()
                    fill = torch.cat([fill_labels[:, None], fill.long()], dim=1)
                else:
                    fill = torch.nn.functional.pad(fill.long(), (1, 0), 'constant', value=empty_label)
                xyzl = torch.cat([xyzl, fill], dim=0)
        if xyzl.shape[0] > max_points:
            idx = torch.randperm(int(xyzl.shape[0]))[:max_points]
            xyzl = xyzl[idx]

    xyz_label = xyzl[:, 0].long()
    xyz_center = xyzl[:, 1:].long()
    xyz = xyz_center.float()

    query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
    query[:, 0] = 2 * xyz[:, 0].clamp(0, grid_size[0] - 1) / float(grid_size[0] - 1) - 1
    query[:, 1] = 2 * xyz[:, 1].clamp(0, grid_size[1] - 1) / float(grid_size[1] - 1) - 1
    query[:, 2] = 2 * xyz[:, 2].clamp(0, grid_size[2] - 1) / float(grid_size[2] - 1) - 1

    return query, xyz_label, xyz_center


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarImgQuery:
    def __init__(
            self,
            data_path,
            return_len,
            offset,
            imageset='train',
            nusc=None,
            times=5,
            max_points=200000,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.max_points = int(max_points)
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

    def __len__(self):
        return len(self.nusc_infos)*self.times


    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]

        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)

        input_occs = np.stack(occs).astype(np.int64)

        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs).astype(np.int64)


        querys, xyz_labels, xyz_centers = [], [], []
        for iii in range(input_occs.shape[0]):
            query, xyz_label, xyz_center = get_query(input_occs[iii].copy(), max_points=self.max_points)
            querys.append(query)
            xyz_labels.append(xyz_label)
            xyz_centers.append(xyz_center)


        metas = {}
        metas["scene_name"] = scene_name
        for i in range(self.return_len + self.offset):
            one_meta = self.get_image_info(scene_name, idx + i)
            metas.update({i: one_meta})


        try:
            if False:
                for one_time_step in range(scene_len):
                    one_path = f"./vis_train_data/{scene_name}/{one_time_step}"
                    os.makedirs(one_path, exist_ok=True)

                    token = self.nusc_infos[scene_name][one_time_step]['token']
                    label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')

                    shutil.copy(
                        os.path.join(self.data_path,
                            f'{self.output_dataset}/{scene_name}/{token}/labels.npz'),
                        one_path)
                    for one_img_file in self.get_image_info(scene_name, one_time_step)['img_filename']:
                        shutil.copy(one_img_file, one_path)
        except:
            pass

        return input_occs[:self.return_len], output_occs[self.offset:], \
                querys, xyz_labels, xyz_centers, metas


    def get_image_info(self, scene_name, idx):
        T = 6
        info = self.nusc_infos[scene_name][idx]

        input_dict = dict(
            time_step=idx,
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )

        if idx != 0:
            cur = np.array(info['ego2global_translation']+info['ego2global_rotation'])
            info_1 = self.nusc_infos[scene_name][idx-1]
            pre = np.array(info_1['ego2global_translation']+info_1['ego2global_rotation'])

            input_dict.update(
                {"traj": cur-pre}
            )
        else:
            input_dict.update(
                {"traj": np.zeros(shape=(7,))}
            )


        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        cam_rotations = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            rotation = R.from_matrix(ego2cam_r)
            euler_angles = rotation.as_euler('xyz', degrees=True)

            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t


            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            cam_rotations.append(euler_angles)
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                cam_rotations=cam_rotations,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))


        return input_dict


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarQuery:
    def __init__(
            self,
            data_path,
            return_len,
            offset,
            imageset='train',
            nusc=None,
            times=5,
            max_points=200000,
            test_mode=False,
            full_length=False,
            flip=True,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.flip = flip
        self.full_length = full_length
        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.max_points = int(max_points)
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

    def __len__(self):
        return len(self.nusc_infos)*self.times


    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]
        if self.full_length:
            idx = 0
            self.return_len = scene_len
            self.offset = 0
        else:
            idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)

        input_occs = np.stack(occs).astype(np.int64)

        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs).astype(np.int64)


        if self.flip:
            p = torch.randint(0, 6, (1,)).item()
            if p == 1:
                input_occs = np.flip(input_occs, axis=1)
                output_occs = np.flip(output_occs, axis=1)
            elif p == 2:
                input_occs = np.flip(input_occs, axis=2)
                output_occs = np.flip(output_occs, axis=2)
            elif p == 3:
                input_occs = np.flip(input_occs, axis=0)
                input_occs = np.flip(input_occs, axis=1)
                output_occs = np.flip(output_occs, axis=0)
                output_occs = np.flip(output_occs, axis=1)


        querys, xyz_labels, xyz_centers = [], [], []
        for iii in range(input_occs.shape[0]):
            query, xyz_label, xyz_center = get_query(input_occs[iii].copy(), max_points=self.max_points)
            querys.append(query)
            xyz_labels.append(xyz_label)
            xyz_centers.append(xyz_center)


        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_length=scene_len)
        metas.update(scene_token=[self.nusc_infos[scene_name][idx + i]['token'] for i in range(self.return_len + self.offset)])
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        metas.update(self.get_meta_data(scene_name, idx))

        for iii in range(self.return_len + self.offset):
            one_info = self.nusc_infos[scene_name][idx + iii]
            one_cams = []
            for cam_type, cam_info in one_info['cams'].items():
                one_cams.append(cam_info['data_path'])
            metas.update({iii: one_cams})

        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))

        try:
            if False:
                for one_time_step in range(scene_len):
                    one_path = f"./vis_train_data/{scene_name}/{one_time_step}"
                    os.makedirs(one_path, exist_ok=True)

                    token = self.nusc_infos[scene_name][one_time_step]['token']
                    label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')

                    shutil.copy(
                        os.path.join(self.data_path,
                            f'{self.output_dataset}/{scene_name}/{token}/labels.npz'),
                        one_path)
                    for one_img_file in self.get_image_info(scene_name, one_time_step)['img_filename']:
                        shutil.copy(one_img_file, one_path)
        except:
            pass

        return input_occs[:self.return_len], output_occs[self.offset:], \
                querys, xyz_labels, xyz_centers, metas


    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0])
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t


            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))

        return input_dict


@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidar:
    def __init__(
            self,
            data_path,
            return_len,
            offset,
            imageset='train',
            nusc=None,
            times=5,
            test_mode=False,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.nusc_infos = data['infos']
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset

    def __len__(self):
        return len(self.nusc_infos)*self.times

    def __getitem__(self, index):
        index = index % len(self.nusc_infos)
        scene_name = self.scene_names[index]
        scene_len = self.scene_lens[index]

        idx = np.random.randint(0, scene_len - self.return_len - self.offset + 1)
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)

        input_occs = np.stack(occs).astype(np.int64)

        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs).astype(np.int64)

        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=[self.nusc_infos[scene_name][idx + i]['token'] for i in range(self.return_len + self.offset)])
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        metas.update(self.get_meta_data(scene_name, idx))


        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))

        try:
            if False:
                for one_time_step in range(scene_len):
                    one_path = f"./vis_train_data/{scene_name}/{one_time_step}"
                    os.makedirs(one_path, exist_ok=True)

                    token = self.nusc_infos[scene_name][one_time_step]['token']
                    label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')

                    shutil.copy(
                        os.path.join(self.data_path,
                            f'{self.output_dataset}/{scene_name}/{token}/labels.npz'),
                        one_path)
                    for one_img_file in self.get_image_info(scene_name, one_time_step)['img_filename']:
                        shutil.copy(one_img_file, one_path)
        except:
            pass

        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_data(self, scene_name, idx):
        gt_modes = []
        xys = []
        for i in range(self.return_len + self.offset):
            xys.append(self.nusc_infos[scene_name][idx+i]['gt_ego_fut_trajs'][0])
            gt_modes.append(self.nusc_infos[scene_name][idx+i]['pose_mode'])
        xys = np.asarray(xys)
        gt_modes = np.asarray(gt_modes)
        return {'rel_poses': xys, 'gt_mode': gt_modes}

    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t


            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_positions.append(focal_position.flatten()[:3])


        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))

        return input_dict

@OPENOCC_DATASET.register_module()
class nuScenesSceneDatasetLidarTraverse(nuScenesSceneDatasetLidar):
    def __init__(
        self,
        data_path,
        return_len,
        offset,
        imageset='train',
        nusc=None,
        times=1,
        test_mode=False,
        use_valid_flag=True,
        input_dataset='gts',
        output_dataset='gts',
    ):
        super().__init__(data_path, return_len, offset, imageset, nusc, times, test_mode, input_dataset, output_dataset)
        self.scene_lens = [l - self.return_len - self.offset for l in self.scene_lens]
        self.use_valid_flag = use_valid_flag
        self.CLASSES = [
            'noise', 'animal' ,'human.pedestrian.adult', 'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
            'movable_object.barrier', 'movable_object.debris',
            'movable_object.pushable_pullable', 'movable_object.trafficcone',
            'static_object.bicycle_rack', 'vehicle.bicycle',
            'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car',
            'vehicle.construction', 'vehicle.emergency.ambulance',
            'vehicle.emergency.police', 'vehicle.motorcycle',
            'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
            'flat.other', 'flat.sidewalk', 'flat.terrain', 'flat.traffic_marking',
            'static.manmade', 'static.other', 'static.vegetation',
            'vehicle.ego'
        ]
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

    def __len__(self):
        return sum(self.scene_lens)

    def __getitem__(self, index):
        for i, scene_len in enumerate(self.scene_lens):
            if index < scene_len:
                scene_name = self.scene_names[i]
                idx = index
                break
            else:
                index -= scene_len
        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.input_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        input_occs = np.stack(occs).astype(np.int64)

        occs = []
        for i in range(self.return_len + self.offset):
            token = self.nusc_infos[scene_name][idx + i]['token']
            label_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/labels.npz')
            label = np.load(label_file)
            occ = label['semantics']
            occs.append(occ)
        output_occs = np.stack(occs).astype(np.int64)

        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=self.nusc_infos[scene_name][4]['token'])
        metas.update(self.get_meta_data(scene_name, idx))
        if self.test_mode:
            metas.update(self.get_meta_info(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        return input_occs[:self.return_len], output_occs[self.offset:], metas

    def get_meta_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        fut_valid_flag = info['valid_flag']
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        if self.with_attr:
            gt_fut_trajs = info['gt_agent_fut_trajs'][mask]
            gt_fut_masks = info['gt_agent_fut_masks'][mask]
            gt_fut_goal = info['gt_agent_fut_goal'][mask]
            gt_lcf_feat = info['gt_agent_lcf_feat'][mask]
            gt_fut_yaw = info['gt_agent_fut_yaw'][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels,
            fut_valid_flag=fut_valid_flag,)

        return anns_results


    def get_image_info(self, scene_name, idx):
        T = 6
        idx = idx + self.return_len + self.offset - 1 - T
        info = self.nusc_infos[scene_name][idx]
        input_dict = dict(
            sample_idx=info['token'],
            ego2global_translation = info['ego2global_translation'],
            ego2global_rotation = info['ego2global_rotation'],
        )
        f = 0.0055
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []

        lidar2ego_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = lidar2ego_r
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation']).T
        ego2lidar = np.linalg.inv(lidar2ego)
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)
            ego2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix)
            ego2cam_t = cam_info['sensor2ego_translation'] @ ego2cam_r.T
            ego2cam_rt = np.eye(4)
            ego2cam_rt[:3, :3] = ego2cam_r.T
            ego2cam_rt[3, :3] = -ego2cam_t


            cam_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            focal_position = np.linalg.inv(ego2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_positions.append(focal_position.flatten()[:3])


        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                ego2lidar=ego2lidar,
                cam_positions=cam_positions,
                focal_positions=focal_positions,
                lidar2ego=lidar2ego,
            ))

        return input_dict
