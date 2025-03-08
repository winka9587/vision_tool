import os
import sys

import numpy as np
import cv2
import pickle
from os.path import join as pjoin
import argparse
from multiprocessing import Process
from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from nocs_utils import backproject, remove_border
from align_pose import pose_fit
"""
    最终不使用这个文件, 修改CAPTRA代码, 直接读取_label.pkl文件
    因为wild6d没有对应的_coord.png, 所以这里的get_pose函数修改为读取_label.pkl文件

"""

# 获取位姿存入一个字典
# key为实例序号,总共有num_instances个实例，从1开始计数
# value为pose_fit得到的位姿
def get_image_pose(num_instances, mask, label_file):
    pose_dict = {}
    
    # 读取pkl文件
    
    for i in range(1, num_instances + 1):
        # mask上的像素点少于3个说明没有得到该物体
        if np.sum(mask == i) < 3:
            continue
        # # 将深度图反投影到3D平面
        # # pts为3D点，idxs为对应的2D坐标，可以使用它去coord中提取对应的点
        # pts, idxs = backproject(depth, intrinsics, mask == i)
        # coord_pts = coord[idxs[0], idxs[1], :]  # already centered
        # if len(pts) < 3:
        #     continue
        # # plot3d_pts([[pts], [coord_pts]])
        # # 深度得到的3D坐标和NOCS的3D坐标进行位姿拟合
        # pose = pose_fit(coord_pts, pts)
        #
        
        if pose is not None:
            pose_dict[i] = pose

    return pose_dict


# root_path=nocs_data/nocs_full/train,val,real_train,real_test
# folders中存储root_path下的实例文件夹
def get_pose(root_path, folders):
    # sub_folder是一组实例文件夹
    for sub_folder in tqdm(folders):
        # file_path = nocs_data/nocs_full/xx_instance
        # file_path为某个实例文件夹的路径
        file_path = pjoin(root_path, sub_folder)
        if not os.path.isdir(file_path):
            continue
        # os.listdir(file_path)列出实例文件夹下的所有文件(图片文件)
        # 筛选出以color.png结尾的文件，取其开头的四个数
        # 结果存于valid_data数组中
        valid_data = [file[:4] for file in os.listdir(file_path) if file.endswith('color.png')]
        valid_data.sort()
        for prefix in valid_data:
            """
            if real or not os.path.exists(pjoin(file_path, f'{prefix}_composed.png')):
                depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'), -1)
                if len(depth.shape) == 3:
                    depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
                    depth = depth.astype(np.uint16)
            else:
                depth = cv2.imread(pjoin(file_path, f'{prefix}_composed.png'), -1)
            """
            # 如果pkl已经存在了,则跳过
            if os.path.exists(pjoin(file_path, f'{prefix}_pose.pkl')) and\
                    os.path.exists(pjoin(file_path, f'{prefix}_meta.txt')):
                continue
            else:
                depth = cv2.imread(pjoin(file_path, f'{prefix}_depth.png'), -1)
                coord = cv2.imread(pjoin(file_path, f'{prefix}_coord.png'))
                mask = cv2.imread(pjoin(file_path, f'{prefix}_mask.png'))
                if depth is None or coord is None or mask is None:
                    print(pjoin(file_path, f'{prefix}_depth.png'))
                    continue
                if len(depth.shape) == 3:
                    depth = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
                    depth = depth.astype(np.uint16)
                mask = mask[:, :, 2]

                # val,train数据集 flip=True
                if flip:
                    depth, coord, mask = depth[:, ::-1], coord[:, ::-1], mask[:, ::-1]

                # 如果是真实数据集,将mask的轮廓一圈删除
                if real:
                    mask = remove_border(mask, kernel_size=2)

                # plot_images([coord, mask])
                # 相当于调换了NOCS map中xyz的值
                # 并使其范围在[-0.5, 0.5]
                coord = coord[:, :, (2, 1, 0)]
                coord = coord / 255. - 0.5
                if not flip:
                    coord[..., 2] = -coord[..., 2]   # verify!!!

                # meta.txt中，每一行对应一个物体
                with open(pjoin(file_path, f'{prefix}_meta.txt'), 'r') as f:
                    lines = f.readlines()
                poses = get_image_pose(len(lines), mask, coord, depth, intrinsics)
                with open(pjoin(file_path, f'{prefix}_pose.pkl'), 'wb') as f:
                    pickle.dump(poses, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    root_path = pjoin(args.data_path, args.data_type)
    # root_path=nocs_data/nocs_full/train,val,real_train,real_test
    folders = os.listdir(root_path)
    # 列出所有实例，存在folders中
    folders.sort()

    if not args.parallel:
        # root_path=nocs_data/nocs_full/train,val,real_train,real_test
        # folders中存储root_path下的实例文件夹
        get_pose(root_path, folders)
    else:
        # 多进程并行处理,理解时和上面的get_pose一样,阅读时直接看上面就行
        # 多进程处理num_per_proc个实例文件夹
        processes = []
        proc_cnt = args.num_proc
        num_per_proc = int((len(folders) - 1) / proc_cnt) + 1

        for k in range(proc_cnt):
            # s_ind(start_index)
            # s_ind和e_ind划分了每个进程处理那几个实例的目录
            s_ind = num_per_proc * k
            e_ind = min(num_per_proc * (k + 1), len(folders))
            p = Process(target=get_pose,
                        args=(root_path, folders[s_ind: e_ind]))
            processes.append(p)
            p.start()

        """
        for process in processes:
            process.join()
        """


if __name__ == '__main__':
    args = parse_args()
    print('get_gt_poses')
    print(args)
    main(args)

