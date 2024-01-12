import os
import sys

import numpy as np
import cv2
from os.path import join as pjoin
import argparse
from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..', '..', '..'))

from CAPTRA_nocs_utils import ensure_dirs

# root_path是nocs_full/train或val或xxx
# folders是数组,保存root_path路径下的所有文件的文件名(00000)
# 如果是真实数据集real=True
# min_points
def get_valid_instance(root_path, folders, real=True, min_points=50):
    # 创建一个字典,然后继续为每一个类创建一个子字典
    # {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
    data_list = {cls_id: {} for cls_id in range(1, 7)}
    # 遍历train/val/real_train/real_val下的所有子文件夹
    for sub_folder in tqdm(folders):
        # file_path的路径是.../nocs_full/train/00000的格式
        file_path = pjoin(root_path, sub_folder)
        if not os.path.isdir(file_path):
            continue
        # valid_data保存file_path目录下所有文件的前缀
        valid_data = [file[:4] for file in os.listdir(file_path) if file.endswith('color.png')]
        valid_data.sort()
        for prefix in valid_data:
            # 读取mask和meta文件
            # filter instances: should belong to class > 0, should appear in mask w/ at least 50 pixels
            mask_path = pjoin(file_path, f'{prefix}_mask.png')
            meta_path = pjoin(file_path, f'{prefix}_meta.txt')
            if not os.path.exists(mask_path) or not os.path.exists(meta_path):
                # print(mask_path, 'does not exist')
                continue
            mask = cv2.imread(mask_path)[:, :, 2]
            # 读取_meta.txt文件
            # appeared_ins = list(np.unique(mask))
            with open(pjoin(meta_path), 'r') as f:
                lines = f.readlines()
            for line in lines:
                if real:
                    # 格式: 1 6 mug2_scene3_norm
                    # 实例序号,类id,实例id,
                    inst_num, cls_id, inst_id = line.split()[:3]
                    # 不知道有什么作用，mug2_scene3_norm并没有发生变化
                    inst_id = inst_id.split('.')[0].replace('/', '_')
                else:
                    # 格式: 1 0 02954340 90c6bffdc81cedbeb80102c6e0a7618a
                    # 实例序号,类id,类code?,实例id
                    inst_num, cls_id, cls_code, inst_id = line.split()[:4]
                inst_num, cls_id = int(inst_num), int(cls_id)
                cnt = np.sum(mask == inst_num)
                # mask中至少要有min_points个点
                if cls_id == 0 or cnt < min_points:
                    continue
                # 如果不在对应class的字典中,将其添加进去(创建一个子字典)
                if inst_id not in data_list[cls_id]:
                    data_list[cls_id][inst_id] = []
                # 将包含前缀的路径保存
                data_list[cls_id][inst_id].append(f'{sub_folder}/{prefix}')

    return data_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../nocs_data')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--list_path', type=str, default='../../nocs_data/instance_list')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_proc', type=int, default=10)

    return parser.parse_args()


def main(args):
    root_path = pjoin(args.data_path, args.data_type)
    # 得到nocs_full/train或val下的所有文件名
    folders = os.listdir(root_path)  # [folder for folder in os.listdir(root_path) if os.path.isdir(pjoin(root_path, folder))]
    folders.sort()
    output_path = pjoin(args.list_path, args.data_type)
    ensure_dirs(output_path)

    # 获取所有可用实例
    data_list = get_valid_instance(root_path, folders)
    # data_list结构
    # data_list{
    #   1:{
    #       instance_1:{
    #           00000/0001
    #           00000/0004
    #           ...
    #       }
    #       instance_2 ...
    #       ...
    #   }
    #   2:{
    #       instance_1 ...
    #       instance_2 ...
    #       ...
    #   }
    #   ...
    # }

    # 在nocs_data\instance_list\data_type目录下,为每个类创建一个目录,1,2,3,4,5,6共6个
    for cls_id in data_list:
        cur_path = pjoin(output_path, str(cls_id))
        ensure_dirs(cur_path)
        for instance_id in data_list[cls_id]:
            # 为每个实例创建一个txt,并保存其包含前缀的路径
            with open(pjoin(cur_path, f'{instance_id}.txt'), 'w') as f:
                for line in data_list[cls_id][instance_id]:
                    print(line, file=f)


if __name__ == '__main__':
    args = parse_args()
    print('get_instance_list')
    print(args)
    main(args)

