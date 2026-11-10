"""Standalone copies of the released occupancy query sampler.

Extracted from dataset/dataset_hexplane.py so training does not import the optional
MMDetection3D/camera dataset stack. Sampling logic and endpoint normalization are
unchanged. Original contributions are covered by the repository Apache-2.0 LICENSE.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

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
