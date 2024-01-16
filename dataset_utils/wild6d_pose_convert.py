# 转换wild6d数据集的pose到CAMERA数据集类型的单帧pkl文件

import sys
from unicodedata import category
import _pickle as cPickle
import numpy as np
import os
import cv2
import json

from nocs_utils import find_bounding_box_2d, draw_and_show_bbox


if len(sys.argv) != 4:
    print("Usage: python script.py arg1 arg2")
    sys.exit(1)

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]  # origin wild6d dataset path, like: /data4/cxx/dataset/Wild6D/test_set/

print("load pkl file from ".format(arg1))
print("save pkl to ".format(arg2))

cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

pkl_path = arg1
# 读取pkl文件，转换为单帧的pkl文件
with open(pkl_path, 'rb') as f:
    result = cPickle.load(f)
    for i in range(result['num_frames']):
        # 保存到新的单个pkl文件
        pass
        new_file = {}
        new_file['instance_ids'] = [1]
        new_file['class_ids'] = [result['annotations'][i]['class_id']]
        new_file['scales'] = np.array([1.0])
        new_file['size'] = result['annotations'][i]['size'][np.newaxis, ...].astype(np.float32)
        new_file['rotations'] = result['annotations'][i]['rotation'][np.newaxis, ...].astype(np.float32)
        new_file['translations'] = result['annotations'][i]['translation'][np.newaxis, ...].astype(np.float32)
        formatted_number = f"{i:04d}_mask.png"
        full_path = os.path.join(arg2, formatted_number)
        # 读取mask, 获得其bboxes
        mask = cv2.imread(full_path)
        y1, x1, y2, x2 = find_bounding_box_2d(mask)
        new_file['model_list'] = [1]
        new_file['bboxes'] = np.array([[y1, x1, y2, x2]])  #  rmin, rmax, cmin, cmax  (y对应r, x对应c)
        new_file['gt_handle_visibility'] = np.array([1])
        
        sRT = np.identity(4, dtype=np.float32)
        sRT[:3, :3] = result['annotations'][i]['rotation']
        sRT[:3, 3] = result['annotations'][i]['translation']
        new_file['poses'] = sRT[np.newaxis, ...].astype(np.float32)
        
        
        pkl_path = full_path.replace("_mask.png", "_label.pkl")
        metatxt_path = full_path.replace("_mask.png", "_meta.txt")
        # 创建_meta.txt文件
        # 并且将内参保存到pkl文件中
        file_path = result['annotations'][0]['name']
        category_name = file_path[:file_path.find('/')]
        cat_id = cat_names.index(category_name) + 1
        # meta_txt_ctx = "1 {} {}".format(cat_id, category_name+"_wild6d")
        meta_txt_ctx = "1 {} {}".format(cat_id, arg2.split('/')[-1].split('\\')[-1]+"_instancemodelwild6d")
        with open(metatxt_path, 'w') as meta_txt_file:
            meta_txt_file.write(meta_txt_ctx)
            meta_txt_file.close()
            print("save _meta.txt file to {}".format(metatxt_path))
        
        # 读取metadata, 保存其内参和图像尺寸到pkl文件
        meta_path = os.path.join(arg3, file_path[:file_path.rfind('/')], 'metadata')  # 获取meta路径
        meta_data = json.load(open(meta_path))
        K = np.array(meta_data['K']).reshape(3, 3).T
        # K = result['intrinsics']  # 可以直接从pkl文件中读取内参K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        new_file['K'] = [fx, fy, cx, cy]
        new_file['h'] = meta_data['h']
        new_file['w'] = meta_data['w']
        
        # 保存到新的pkl文件
        with open(pkl_path, 'wb') as pkl_file:
            cPickle.dump(new_file, pkl_file)
            print("save _label.pkl file to {}".format(pkl_path))

"""
NOCS
    class_ids √
    bboxes (需要重新分割) √
    scales √
    rotations √
    translations √
    instance_ids √
    model_list (mug会需要使用) 暂时不管
    
    # 可能还需要额外保存相机的内参
    # 因为整个test序列不同于REAL275相机不变

wild6d pkl:
    name
    class_id
    rotation
    size
    translation
"""
