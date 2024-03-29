import numpy as np
import torch
import os

def backproject(depth, intrinsics=np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]]),
                mask=None, scale=0.001):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    non_zero_mask = (depth > 0)
    # mask与深度不为0的点进行逻辑并运算
    if mask is not None:
        final_instance_mask = np.logical_and(mask, non_zero_mask)
    else:
        final_instance_mask = non_zero_mask

    # where返回为True的索引
    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], height - idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    # 将idx对应索引的点都拿出来
    z = depth[idxs[0], idxs[1]].astype(np.float32)

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    pts[:, 2] = -pts[:, 2]   # x, y is divided by |z| during projection --> here depth > 0 = |z| = -z

    return pts * scale, idxs

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def project(pts, intrinsics, scale=1000):  # not flipping y axis
    pts = pts * scale
    pts = -pts / pts[:, -1:]
    pts[:, -1] = -pts[:, -1]
    pts = np.transpose(intrinsics @ np.transpose(pts))[:, :2]
    return pts


def remove_border(mask, kernel_size=2):  # enlarge the region w/ 255
    # print((255 - mask).sum())
    output = mask.copy()
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i][j] == 255:
                output[max(0, i - kernel_size): min(h, i + kernel_size),
                max(0, j - kernel_size): min(w, j + kernel_size)] = 255
    # print((255 - output).sum())
    return output

# 输入:[中心点减半径,中心点加半径]
#
def get_corners(points):  # [Bs, N, 3] -> [Bs, 2, 3]
    if isinstance(points, torch.Tensor):
        points = np.array(points.detach().cpu())
    pmin = np.min(points, axis=-2) # [Bs, N, 3] -> [Bs, 3]
    pmax = np.max(points, axis=-2) # [Bs, N, 3] -> [Bs, 3]
    return np.stack([pmin, pmax], axis=-2)


def bbox_from_corners(corners):  # corners [[3], [3]] or [Bs, 2, 3]
    # 确保corners是n维数组
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners)

    # bbox = np.zeros((8, 3))
    bbox_shape = corners.shape[:-2] + (8, 3)  # [Bs, 8, 3]
    bbox = np.zeros(bbox_shape)
    for i in range(8):
        x, y, z = (i % 4) // 2, i // 4, i % 2
        bbox[..., i, 0] = corners[..., x, 0]
        bbox[..., i, 1] = corners[..., y, 1]
        bbox[..., i, 2] = corners[..., z, 2]
    return bbox

def ensure_dir(path, verbose=False):
    if not os.path.exists(path):
        if verbose:
            print("Create folder ", path)
        os.makedirs(path)
    else:
        if verbose:
            print(path, " already exists.")


def ensure_dirs(paths):
    if isinstance(paths, list):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)