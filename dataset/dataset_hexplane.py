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
    sampling_strategy: str = 'balanced',
    empty_label: int = 17,
    boundary_distance: int = 2,
    occupied_ratio: float = 0.9,
):
    labels = torch.from_numpy(np.asarray(voxel_label)).long()
    sampling_strategy = str(sampling_strategy).lower()
    empty_label = int(empty_label)
    num_class = int(num_class)

    if sampling_strategy == 'random':
        grid_x, grid_y, grid_z = [int(v) for v in labels.shape]
        total_voxels = int(grid_x * grid_y * grid_z)
        sample_count = min(int(max_points), total_voxels)
        flat_idx = torch.randperm(total_voxels)[:sample_count]
        x = torch.div(flat_idx, grid_y * grid_z, rounding_mode='floor')
        yz = flat_idx % (grid_y * grid_z)
        y = torch.div(yz, grid_z, rounding_mode='floor')
        z = yz % grid_z
        xyz_center = torch.stack([x, y, z], dim=1).long()
        xyz_label = labels[xyz_center[:, 0], xyz_center[:, 1], xyz_center[:, 2]].long()
        xyz = xyz_center.float()

        query = torch.zeros(xyz.shape, dtype=torch.float32, device=xyz.device)
        query[:, 0] = 2 * xyz[:, 0].clamp(0, grid_size[0] - 1) / float(grid_size[0] - 1) - 1
        query[:, 1] = 2 * xyz[:, 1].clamp(0, grid_size[1] - 1) / float(grid_size[1] - 1) - 1
        query[:, 2] = 2 * xyz[:, 2].clamp(0, grid_size[2] - 1) / float(grid_size[2] - 1) - 1
        return query, xyz_label, xyz_center

    if sampling_strategy != 'balanced':
        raise ValueError(f'unsupported sampling_strategy: {sampling_strategy}')

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
class nuScenesSceneDatasetLidarHexPlane:
    def __init__(
            self,
            data_path,
            return_len,
            offset,
            imageset='train',
            nusc=None,
            times=1,
            max_points=200000,
            sampling_strategy='balanced',
            test_mode=False,
            use_valid_flag=True,
            flip=True,
            input_dataset='gts',
            output_dataset='gts'
        ):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.window_infos = data.get('windows', None)
        self.scene_names = list(self.nusc_infos.keys())
        self.scene_lens = [len(self.nusc_infos[sn]) for sn in self.scene_names]
        self.data_path = data_path
        self.return_len = return_len
        self.use_valid_flag = use_valid_flag
        self.offset = offset
        self.nusc = nusc
        self.times = times
        self.test_mode = test_mode
        self.flip = flip
        assert input_dataset in ['gts', 'tpv_dense', 'tpv_sparse']
        assert output_dataset == 'gts', f'only used for evaluation, output_dataset should be gts, but got {output_dataset}'
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.with_velocity = True
        self.with_attr = True
        self.box_mode_3d = Box3DMode.LIDAR

        self.max_points = max_points
        self.sampling_strategy = str(sampling_strategy).lower()

    def __len__(self):

        if self.window_infos is not None:
            return len(self.window_infos)
        return len(self.nusc_infos)


    def __getitem__(self, index):
        if self.window_infos is not None:
            window = self.window_infos[index % len(self.window_infos)]
            scene_name = window['scene_name']
            idx = int(window['start_idx'])
            scene_len = len(self.nusc_infos[scene_name])
        else:
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

        if self.flip:
            p = torch.randint(0, 6, (1,)).item()
            if p == 1:
                input_occs = np.flip(input_occs, axis=1).copy()
                output_occs = np.flip(output_occs, axis=1).copy()
            elif p == 2:
                input_occs = np.flip(input_occs, axis=2).copy()
                output_occs = np.flip(output_occs, axis=2).copy()
            elif p == 3:
                input_occs = np.flip(input_occs, axis=0).copy()
                input_occs = np.flip(input_occs, axis=1).copy()
                output_occs = np.flip(output_occs, axis=0).copy()
                output_occs = np.flip(output_occs, axis=1).copy()

        triplanes = []
        for iii in range(input_occs.shape[0]):
            token = self.nusc_infos[scene_name][idx + iii]['token']
            triplane_file = os.path.join(self.data_path, f'{self.output_dataset}/{scene_name}/{token}/triplane.pth')
            triplane = torch.load(triplane_file, map_location="cpu")
            triplanes.append(triplane)

        querys, xyz_labels, xyz_centers = [], [], []
        for iii in range(input_occs.shape[0]):
            query, xyz_label, xyz_center = get_query(
                input_occs[iii].copy(),
                max_points=self.max_points,
                sampling_strategy=self.sampling_strategy,
            )
            querys.append(query)
            xyz_labels.append(xyz_label)
            xyz_centers.append(xyz_center)


        metas = {}
        metas.update(scene_name=scene_name)
        metas.update(scene_token=[self.nusc_infos[scene_name][idx + i]['token'] for i in range(self.return_len + self.offset)])
        metas.update(self.get_meta_data(scene_name, idx))
        metas.update(self.get_image_info(scene_name,idx))
        metas.update(self.get_meta_data(scene_name, idx))

        ego2global_translation_seq = np.asarray(
            [self.nusc_infos[scene_name][idx + i]['ego2global_translation'] for i in range(self.return_len + self.offset)],
            dtype=np.float32,
        )
        ego2global_rotation_seq = np.asarray(
            [self.nusc_infos[scene_name][idx + i]['ego2global_rotation'] for i in range(self.return_len + self.offset)],
            dtype=np.float32,
        )
        can_bus_seq = np.asarray(
            [self.nusc_infos[scene_name][idx + i]['can_bus'] for i in range(self.return_len + self.offset)],
            dtype=np.float32,
        )
        gt_ego_fut_cmd_seq = np.asarray(
            [self.nusc_infos[scene_name][idx + i]['gt_ego_fut_cmd'] for i in range(self.return_len + self.offset)],
            dtype=np.float32,
        )
        gt_ego_lcf_feat_seq = np.asarray(
            [self.nusc_infos[scene_name][idx + i]['gt_ego_lcf_feat'] for i in range(self.return_len + self.offset)],
            dtype=np.float32,
        )
        metas.update(ego2global_translation_seq=ego2global_translation_seq)
        metas.update(ego2global_rotation_seq=ego2global_rotation_seq)
        metas.update(can_bus_seq=can_bus_seq)
        metas.update(gt_ego_fut_cmd_seq=gt_ego_fut_cmd_seq)
        metas.update(gt_ego_lcf_feat_seq=gt_ego_lcf_feat_seq)

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

        return input_occs[:self.return_len], output_occs[self.offset:], triplanes, \
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
