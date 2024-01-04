# 转换wild6d数据集的pose到CAMERA数据集类型的单帧pkl文件

import sys
import _pickle as cPickle


if len(sys.argv) != 3:
    print("Usage: python script.py arg1 arg2")
    sys.exit(1)

arg1 = sys.argv[1]
arg2 = sys.argv[2]

print("load pkl file from ".format(arg1))
print("save pkl to ".format(arg2))

pkl_path2 = "/data1/jl/awsl-JL/object-deformnet-master/object-deformnet-master/data/Real/train/scene_1/0000_label.pkl"
with open(pkl_path2, 'rb') as f2:
    result2 = cPickle.load(f2)  # 测试: 读取用于对比的NOCS的pkl文件

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
        new_file['scales'] = [result['annotations'][i]['scale']]
        new_file['rotations'] = [result['annotations'][i]['rotation']]
        new_file['translations'] = [result['annotations'][i]['translation']]
        
        new_file['model_list'] = [1]
        
        new_file['bboxes'] = [[y1, x1, y2, x2]]  #  rmin, rmax, cmin, cmax  (y对应r, x对应c)
    
    
    class_idx = cat_names.index(opt.select_class) + 1
    result['gt_class_ids'] = np.array([class_idx], dtype=np.int32)
    result['gt_bboxes'] = result['gt_bboxes'][np.newaxis, ...]
    result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
    result['pred_class_ids'] = np.array([class_idx], dtype=np.int32)
    result['pred_scores'] = np.array([result['pred_scores']])

"""
NOCS
    class_ids √
    bboxes (需要重新分割)
    scales √
    rotations √
    translations √
    instance_ids √
    model_list (mug会需要使用)
    
    # 可能还需要额外保存相机的内参
    # 因为整个test序列不同于REAL275相机不变

wild6d pkl:
    name
    class_id
    rotation
    size
    translation

"""
