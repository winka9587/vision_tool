"""
    wild6d 对 CAPTRA的函数替换
    
    1.所有涉及到RT的位姿变换 transform_coordinates_3d, 输入是(3, N)和(4, 4), 输出是(3, N), 在CAPTRA中使用要再加个transpose()变成(N, 3)
    示例:
        scale = label_dict['size'][0]  # _label.pkl文件中的size是nocs的包围盒在三个维度的长度
        rot = label_dict['rotations'][0]
        trans = label_dict['translations'][0]
        RTs = np.eye(4)
        RTs[:3, :3] = rot
        RTs[:3, 3] = trans
        noc_cube = get_3d_bbox(scale, 0)
        posed_bbox = transform_coordinates_3d(noc_cube, RTs).transpose()
    2.投影和反投影
        project 换成 calculate_2d_projections
        backproject 换成 backproject2
        
    3.get_3d_bbox 用于使用 _label.pkl文件中的size, 生成nocs空间下的3d包围盒
    
    4.本文件中所有带有 # from wild6d 的函数都是从wild6d中复制过来的, 用于替换CAPTRA中的函数


"""
import os
import sys

import numpy as np
import cv2
import pickle
from os.path import join as pjoin
import argparse
from multiprocessing import Process

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..', '..'))

from CAPTRA_nocs_utils import backproject, project, get_corners, bbox_from_corners, ensure_dirs
from tqdm import tqdm

# import open3d as o3d
# class PointCloudRender:
#     def __init__(self, result_dir=None, window_shape=(512, 512), window_pos=(50, 25)):
#         self.result_dir = result_dir
#         self.window_width = window_shape[0]
#         self.window_height = window_shape[1]
#         self.left = window_pos[0]
#         self.top = window_pos[1]
#         self.rotate_value = 10.0
#         self.rotate_count = [0]
#         self.coordinate = False
#         self.coordinateMesh = None

#     def __rotate_view(self, vis):
#         """
#         旋转回调函数
#         :param vis:  vis = o3d.visualization.Visualizer()
#         :return:
#         """
#         ctr = vis.get_view_control()
#         # ctr.set_zoom(2)
#         ctr.rotate(self.rotate_value, 0)
#         if self.result_dir and 1 < self.rotate_count[0] < 525:
#             save_path = os.path.join(self.result_dir, f'{self.rotate_count[0]}.png')
#             vis.capture_screen_image(save_path, False)
#         self.rotate_count[0] += 1
#         return False

#     def __capture_image(self, vis):
#         img_save_path = os.path.join(self.result_dir, "{0}.png".format(time.asctime().replace(' ', '-').replace(':', '-')))
#         image_buffer = vis.capture_screen_float_buffer()
#         img_tmp = np.asarray(image_buffer)
#         img = img_tmp.copy()
#         img[:, :, 0] = img_tmp[:, :, 2]
#         img[:, :, 2] = img_tmp[:, :, 0]

#         cv2.imshow("captured image", img)
#         cv2.waitKey(1)
#         if self.result_dir:

#             cv2.imwrite(img_save_path, img * 255.0)
#             print("[OK] img save to {0}".format(img_save_path))
#         return False

#     def open_coordinate(self, open):
#         """
#         是否开启坐标系
#         :param open: True 在可视化中显示坐标系; False 在可视化中关闭坐标系
#         :return:
#         """
#         self.coordinate = open

#     def coordinate_setting(self, scale=1.0, center=(0, 0, 0)):
#         """
#         调整坐标系的scale, 坐标原点位置
#         x, y, z 坐标轴将分别渲染为红色, 绿色, 蓝色
#         :param scale: 坐标系尺寸
#         :param center: tuple(x, y, z) 坐标系原点位置
#         :return:
#         """
#         mesh_ = o3d.geometry.TriangleMesh.create_coordinate_frame()
#         mesh_.scale(scale, center=(0, 0, 0))
#         self.coordinateMesh = mesh_

#     """
#     Args:
#         name: windows name 窗口名
#         pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list
#         color_list: list of color (every color like np.array([255,0,0])) 存放颜色的list，list长度应该与pts_list相同
#         # 修复, color_list 现在可以接收单独定义每个点的颜色(1,3) -> (1, 3) 或 (n, 3)
#         result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
#     """
#     def render_multi_pts(self, win_name, pts_list, color_list, save_dir=None, save_img=False, show_coord=True):
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(window_name=win_name, width=512, height=512, left=300, top=300)
#         opt = vis.get_render_option()
#         opt.show_coordinate_frame = show_coord
#         assert len(pts_list) == len(color_list)
#         pcds = []
#         for index in range(len(pts_list)):
#             print(1)
#             pcd = o3d.geometry.PointCloud()
#             pts = pts_list[index]
#             color = color_list[index]
#             if color.shape != pts.shape:
#                 print("color shape != pts.shape")
#                 colors = np.tile(color, (pts.shape[0], 1))
#             else:
#                 colors = color
#             pcd.points = o3d.utility.Vector3dVector(pts)
#             pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
#             pcds.append(pcd)
#         if self.coordinate:
#             vis.add_geometry(self.coordinateMesh)
#         for pcd in pcds:
#             vis.add_geometry(pcd)
#         ctr = vis.get_view_control()

#         key_to_callback = {}
#         key_to_callback[ord("C")] = self.__capture_image
#         o3d.visualization.draw_geometries_with_key_callbacks([pcd for pcd in pcds], key_to_callback, window_name=win_name,
#                                                                 width=self.window_width, height=self.window_height,
#                                                                 left=self.left, top=self.top)
#         if save_img:
#             img_save_path = os.path.join(save_dir, "{0}.png".format(time.asctime().replace(' ', '-').replace(':', '-')))
#             vis.capture_screen_image(img_save_path, False)
#             print("[OK] img save to {0}".format(img_save_path))


#     def visualize_shape(self, name, pts, result_dir=None):
#         """
#         最简单的可视化函数
#         :param name: 窗口名
#         :param pts: (numpy.array, nx3) 点云
#         :param result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
#         :return:
#         """
#         """ The most simple function, for visualization pointcloud and save image.
#         Args:
#             name: window name 
#             pts: list of pointcloud 
#             result_dir: if not None, save image to this path 图片保存路径

#         """
#         self.result_dir = result_dir
#         vis = o3d.visualization.Visualizer()
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pts)
#         vis.add_geometry(pcd)
#         ctr = vis.get_view_control()

#         key_to_callback = {}
#         key_to_callback[ord("C")] = self.__capture_image
#         o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, window_name=name,
#                                                                 width=self.window_width, height=self.window_height,
#                                                                 left=self.left, top=self.top)


"""
Args:
    name: windows name 窗口名
    pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list
    color_list: list of color (every color like np.array([255,0,0])) 存放颜色的list，list长度应该与pts_list相同
    # 修复, color_list 现在可以接收单独定义每个点的颜色(1,3) -> (1, 3) 或 (n, 3)
    result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
"""
def render_multi_pts(self, win_name, pts_list, color_list, save_dir=None, save_img=False, show_coord=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=win_name, width=512, height=512, left=300, top=300)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = show_coord
    assert len(pts_list) == len(color_list)
    pcds = []
    for index in range(len(pts_list)):
        print(1)
        pcd = o3d.geometry.PointCloud()
        pts = pts_list[index]
        color = color_list[index]
        if color.shape != pts.shape:
            print("color shape != pts.shape")
            colors = np.tile(color, (pts.shape[0], 1))
        else:
            print("color shape == pts.shape")
            colors = color
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # pcds.append(pcd)
        vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.rotate(-300.0, 150.0)
    vis.run()
    if save_img:
        vis.capture_screen_image(os.path.join(save_dir, name + ',png'), False)
    vis.destroy_window()


def render_multi_pts_rotation(self, name, pts_list, color_list,
                                rotate_value=8.0, angle_offset=(0, 0, 0), result_dir=None):
    """
    旋转可视化点云, 自动截图并保存到指定目录
    :param name:
    :param pts_list: list of pointcloud (numpy.array, nx3) 点云
    :param color_list: 点云颜色 例如[np.array([255,0,0]), np.array([255,255,0])], 此参数如果不为空, 长度必须和pts_list一致
    :param rotate_value: 该值的大小控制每次旋转的幅度 默认8.0
    :param angle_offset: 调整点云绕xyz轴的初始旋转。(值的范围为-2~2,对应-2*PI~2*PI) (numpy.float)
    :param result_dir: 存储图片的路径, 如果不提供则不会保存 例如: "/data/cat"
    :return:
    """
    self.rotate_value = rotate_value
    self.result_dir = result_dir
    vis = o3d.visualization.Visualizer()
    assert len(pts_list) == len(color_list)

    pcds = []
    for index in range(len(pts_list)):
        pcd = o3d.geometry.PointCloud()
        pts = pts_list[index]
        color = color_list[index]
        if len(color.shape) == 1 or color.shape[0] != pts.shape[0]:
            if np.any(color > 1.0):
                color = color.astype(float) / 255
            color = np.tile(color, (pts.shape[0], 1))  # 将color扩大为 (pts.shape[0], 1)
        else:
            print("Error in viz_pts_with_rotation: color shape {0} wrong".format(color.shape))
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # 调整pts角度
        x_r, y_r, z_r = angle_offset
        R = pcd.get_rotation_matrix_from_xyz((x_r * np.pi, y_r, z_r * np.pi))
        pcd = pcd.rotate(R, center=(0, 0, 0))

        pcds.append(pcd)
        vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    o3d.visualization.draw_geometries_with_animation_callback(pcds, self.__rotate_view, window_name=name,
                                                            width=self.window_width, height=self.window_height,
                                                            left=self.left, top=self.top)
    self.rotate_count[0] = 0

# from wild6d
def transform_coordinates_3d(coordinates, sRT):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = sRT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

# from wild6d
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

# from wild6d
def backproject2(depth, intrinsics, instance_mask=None, scale=1000.0):
    """ Back-projection, use opencv camera coordinate frame.

    """
    cam_fx = intrinsics[0, 0]
    cam_fy = intrinsics[1, 1]
    cam_cx = intrinsics[0, 2]
    cam_cy = intrinsics[1, 2]

    non_zero_mask = (depth > 0)
    if instance_mask is not None:
        final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    else:
        final_instance_mask = non_zero_mask
    idxs = np.where(final_instance_mask)

    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - cam_cx) * z / cam_fx
    y = (idxs[0] - cam_cy) * z / cam_fy
    pts = np.stack((x, y, z), axis=1)
    pts = pts/scale
    return pts, idxs
    """
    # CAPTRA的backproject结果
    raw_pts
    array([[-0.25467258,  0.3217909 , -0.458     ],
        [-0.25417398,  0.3224935 , -0.459     ],
        [-0.25311933,  0.3224935 , -0.459     ],
        ...,
        [ 0.36446801, -0.49353906, -0.675     ],
        [ 0.37144146, -0.50085075, -0.685     ],
        [ 0.37301539, -0.50085075, -0.685     ]])

    # wild6d的反投影backproject2结果, 没有除以1000
    raw_pts2/1000
    array([[-0.25467258, -0.32014249,  0.458     ],
        [-0.25417398, -0.32084149,  0.459     ],
        [-0.25311933, -0.32084149,  0.459     ],
        ...,
        [ 0.36446801,  0.49596848,  0.675     ],
        [ 0.37144146,  0.50331616,  0.685     ],
        [ 0.37301539,  0.50331616,  0.685     ]])
    """

# from wild6d
def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [3, N]

    """
    bbox_3d = np.array([[+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d


def gather_instances(list_path, data_path, model_path, output_path, instances,
                     flip=True, real=False):
    # 对每一个实例,收集其所有数据
    for instance in tqdm(instances):
        gather_instance(list_path, data_path, model_path, output_path,
                        instance, flip=flip, real=real)


def gather_instance(list_path, data_path, model_path, output_path, instance,
                    flip=True, real=True, img_per_folder=100, render_rgb=False):
    # 获得实例的所有路径
    meta_path = pjoin(list_path, f'{instance}.txt')
    with open(meta_path, 'r') as f:
        lines = f.readlines()

    # nocs_data/instance_data/(instance_id)
    inst_output_path = pjoin(output_path, instance)

    if not real:
        # 如果不是真实数据集,创建路径nocs_data/instance_data/(instance_id)/0000
        folder_num, img_num = 0, -1
        cur_folder_path = pjoin(inst_output_path, f'{folder_num:04d}')
        # render_rgb = False
        # nocs_data/instance_data/(instance_id)/0000/data
        # 如果render_rgb则再创建一个rgb目录
        ensure_dirs([pjoin(cur_folder_path, name) for name in (['data'] if not render_rgb else ['rgb', 'data'])])

    meta_dict = {}

    # (instance_id).txt中的每一行line是一个路径,例如00000/0001
    # 指向train/00000/0001_color.png
    for line in tqdm(lines, desc=f'Instance {instance}'):
        # 00000/0001
        # 00000 是track_name
        # 0001  是prefix
        track_name, prefix = line.strip().split('/')[:2]
        file_path = pjoin(data_path, track_name)
        # 如果是真实数据集,将track_name添加到meta_dict中,值为路径.../nocs_fulll/train/00000
        if real and track_name not in meta_dict:
            meta_dict[track_name] = file_path
        # 如果是真实数据集,后缀为depth
        # 如果是合成数据集,后缀为composed
        # 因为合成数据集需要_composed.png来提供深度信息
        suffix = 'depth'
        try:
            depth = cv2.imread(pjoin(file_path, f'{prefix}_{suffix}.png'), -1)
            mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))[:, :, 2]
            mask[np.where(mask==255)]=False
            mask[np.where(mask==1)]=True
            rgb = cv2.imread(pjoin(file_path, f'{prefix}_color.png'))
            # 读取_meta.txt来获得图中的所有实例
            with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
                meta_lines = f.readlines()
            # # 读取_pose.pkl来获得图中的所有实例的gt位姿
            # # 可以通过pose_dict[i]来获得位姿pose
            with open(pjoin(file_path, f'{prefix}_label.pkl'), 'rb') as f:
                label_dict = pickle.load(f)
            # 验证格式
            # model_path_ = "/data2/cxx/dataset/NOCS/model_corners/1a0a2715462499fbf9029695a3277412.npy"
            # corners2 = np.load(model_path_)
            pose_dict = label_dict['poses']
            
            fx, fy, cx, cy = label_dict['K']
            # intrinsics从pkl中直接读取
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            # 读取/nocs_data/model_corners/instance_id.npy
            # 获得包围盒的8个顶点
            # corners = np.load(pjoin(model_path, f'{instance}.npy'))
            size = label_dict['size']  # wild6d存储的size是长宽高, 在创建3dbbpx时候需要将其除以2
            x,y,z = size[0]/2
            corners = np.array([[[-x, -y, -z], [x, y, z]]])
            bbox = bbox_from_corners(corners[0])
            bbox *= 1.4  # 包围盒大小乘以1.4
        except:
            continue
        # 合成数据集flip为True
        # 第二个维度需要颠倒
        inst_num = 1
        # 从pose_dict中读取gt位姿
        pose = pose_dict[0]
        # bbox的8个点都使用gt位姿进行变换
        # posed_bbox = (np.matmul(bbox, label_dict['rotations'][0].swapaxes(-1, -2))
        #               * np.expand_dims(label_dict['scales'][0], (-1, -2))
        #               + label_dict['translations'][0][np.newaxis, :])  # posed_bbox [8, 3]
        scale = label_dict['size'][0]
        rot = label_dict['rotations'][0]
        trans = label_dict['translations'][0]
        RTs = np.eye(4)
        RTs[:3, :3] = rot
        RTs[:3, 3] = trans
        noc_cube = get_3d_bbox(scale, 0)
        posed_bbox = transform_coordinates_3d(noc_cube, RTs).transpose()
        
        # pcr = PointCloudRender()
        
        # 计算位姿变换后包围盒的中心点(相加平均)
        center = posed_bbox.mean(axis=0)
        # 半径为一个顶点到中心点的距离再加0.1
        # radius = np.sqrt(np.sum((posed_bbox[0] - center) ** 2)) + 0.1
        radius = np.sqrt(np.sum((posed_bbox[0] - center) ** 2))

        # 输入:[中心点减半径,中心点加半径]
        # 输出:np.stack([pmin, pmax] 两个点按大小排列
        aa_corner = get_corners([center - np.ones(3) * radius, center + np.ones(3) * radius])
        # 通过上面的两个点获得新的包围盒aabb
        aabb = bbox_from_corners(aa_corner)

        height, width = label_dict['h'], label_dict['w']
        # 将aabb投影到平面,project返回的是像素坐标系的坐标(u,v)
        # [:, [1,0]]  第一个冒号,取全部点;第二个[1,0]取第1和第0列并交换顺序
        # projected_corners = project(aabb, intrinsics).astype(np.int32)[:, [1, 0]]  # 问题出在: projected_corners 投影后的包围盒顶点是错误的, 导致mask全部是1
        projected_corners = calculate_2d_projections(aabb.transpose(), intrinsics).astype(np.int32)[:, [1, 0]]
        projected_corners[:, 0] = height - projected_corners[:, 0]
        # 得到2D包围盒的2个顶点
        corner_2d = np.stack([np.min(projected_corners, axis=0),
                              np.max(projected_corners, axis=0)], axis=0)
        # 检查这两个点是否在图像外
        corner_2d[0, :] = np.maximum(corner_2d[0, :], 0)
        corner_2d[1, :] = np.minimum(corner_2d[1, :], np.array([height - 1, width - 1]))
        corner_mask = np.zeros_like(mask)
        # 根据这两个顶点绘制一个矩形(矩形内的值为1),创建corner_mask
        corner_mask[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1] = 1
        # 将_color.png对应的矩形区域裁剪出来
        cropped_rgb = rgb[corner_2d[0, 0]: corner_2d[1, 0] + 1, corner_2d[0, 1]: corner_2d[1, 1] + 1]

        # corner_mask与depth取交集
        # 根据2D bbox反投影得到点云
        # raw_pts, raw_idx = backproject(depth, intrinsics=intrinsics, mask=corner_mask)
        # raw_pts, raw_idx = backproject2(depth, intrinsics=intrinsics, instance_mask=corner_mask)
        raw_pts, raw_idx = backproject2(depth, intrinsics=intrinsics, instance_mask=mask)
        # 将2D bbox包含的点和mask包含的点取交集,得到raw_mask
        raw_mask = (mask == inst_num)[raw_idx[0], raw_idx[1]]

        def filter_box(pts, corner):
            mask = np.prod(np.concatenate([pts >= corner[0], pts <= corner[1]], axis=1).astype(np.int8),  # [N, 6]
                           axis=1)
            idx = np.where(mask == 1)[0]
            return pts[idx]

        def filter_ball(pts, center, radius):
            distance = np.sqrt(np.sum((pts - center) ** 2, axis=-1))  # [N]
            idx = np.where(distance <= radius)
            return pts[idx], idx

        # 筛选出raw_pts中再radius为半径的球内的点
        # 这些点再一次对raw_mask进行筛选
        pts, idx = filter_ball(raw_pts, center, radius)
        obj_mask = raw_mask[idx]

        # 将所有信息保存到data_dict中
        # pts是当前实例的点云
        # labels是当前实例点云对应的2D mask, 对应点的值为inst_num,可通过obj_mask[idx]来进行访问
        data_dict = {'points': pts, 'labels': obj_mask, 'pose': pose,
                     'path': pjoin(file_path, f'{prefix}_{suffix}.png')}
        # 合成数据集
        if not real:
            img_num += 1
            # 如果img_num超过限度,则(instance_id)下再创建一个文件夹
            if img_num >= img_per_folder:
                folder_num += 1
                # 例:nocs_data/instance_data/(instance_id)/0001
                cur_folder_path = pjoin(inst_output_path, f'{folder_num:04d}')
                # nocs_data/instance_data/(instance_id)/0001/data
                ensure_dirs([pjoin(cur_folder_path, name) for name in (['data'] if not render_rgb else ['rgb', 'data'])])
                img_num = 0
            # 将data_dict保存至/data/(img_num).npz
            # 例如/data/01.npz
            np.savez_compressed(pjoin(cur_folder_path, 'data', f'{img_num:02d}.npz'), all_dict=data_dict)
            if render_rgb:
                cv2.imwrite(pjoin(cur_folder_path, 'rgb', f'{img_num:02d}.png'), cropped_rgb)
        else:
            cur_folder_path = pjoin(inst_output_path, track_name)
            ensure_dirs(pjoin(cur_folder_path, 'data'))
            np.savez_compressed(pjoin(cur_folder_path, 'data', f'{prefix}.npz'), all_dict=data_dict)

    if real:
        cur_folder_path = pjoin(inst_output_path, track_name)
        ensure_dirs([cur_folder_path])
        for track_name in meta_dict:
            with open(pjoin(cur_folder_path, 'meta.txt'), 'w') as f:
                print(meta_dict[track_name], file=f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--list_path', type=str, default='../../nocs_data/instance_list')
    parser.add_argument('--model_path', type=str, default='../../nocs_data/model_corners')
    parser.add_argument('--output_path', type=str, default='../../nocs_data/instance_data')
    parser.add_argument('--category', type=int, default=1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    data_path = pjoin(args.data_path, args.data_type)  # 数据集路径与数据集类型拼接
    list_path = pjoin(args.list_path, args.data_type, str(args.category))  # get_instance_list.py生成的保存实例路径的目录
    model_path = args.model_path  # 模型所在路径 默认/nocs_data/model_corners
    output_path = pjoin(args.output_path, args.data_type, str(args.category))  # 默认是nocs_data/instance_data
    ensure_dirs(output_path)
    instances = list(map(lambda s: s.split('.')[0], os.listdir(list_path)))  # 去掉.txt 只保留instance这个前缀
    instances.sort()

    if not args.parallel:
        gather_instances(list_path, data_path, model_path, output_path, instances)
    else:
        processes = []
        proc_cnt = args.num_proc
        num_per_proc = int((len(instances) - 1) / proc_cnt) + 1

        for k in range(proc_cnt):
            s_ind = num_per_proc * k
            e_ind = min(num_per_proc * (k + 1), len(instances))
            p = Process(target=gather_instances,
                        args=(list_path, data_path, model_path, output_path,
                              instances[s_ind: e_ind]))
            processes.append(p)
            p.start()

        """
        for process in processes:
            process.join()
        """


if __name__ == '__main__':
    args = parse_args()
    print('get_instance_data')
    print(args)
    main(args)

