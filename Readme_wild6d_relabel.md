
# Wild6D数据集重新组织

### Wild6D testset 结构
```
data
├── testset
│   ├── bottle
|       ├── 0001
|           ├── 1(有些序列为日期, 如: 2021-09-03--10-46-31)
|               ├── images(有些序列为rgbd)
|               ├── metadata (保存fps, h, w, K等信息)
|           ├── 2
|           ├── ...
|       ├── 0002
|       ├── 0003
|       ├── ...
│   ├── bowl
│   ├── camera
│   ├── laptop
│   ├── mug
│   └── pkl_annotations
│       ├── bottle
│           ├── bottle-0001-1.pkl
│           ├── bottle-0001-2.pkl
│           ├── bottle-0001-3.pkl
│           ├── bottle-0002-1.pkl
│           ├── ...
│       ├── bowl
│       ├── camera
│       ├── laptop
│       ├── mug
```

### 组织思路

所需内容
- [x] class_ids 
- [ ] bboxes (需要重新分割) 
- [x] scales 
- [x] rotations 
- [x] translations 
- [x] instance_ids 
- [ ] model_list (mug会使用, 暂时不管)

NOCS dataset的组织如下:

```
├──test
|   ├──scene_1
|       ├──0000_color.png [√]
|       ├──0000_coord.png [×]
|       ├──0000_depth.png [√]
|       ├──0000_mask.png  [×]
|       ├──0000_meta.txt  [×]
|       ├──0000_label.pkl [×]
```

操作:
1. 使用Tracking-Anything为每个序列重新生成_mask.png
2. wild6d_rgbd_relabel.sh调用wild6d_relabel_color.sh与data_relabel_suffix.sh将rgb, depth, mask图像复制到新的保存路径中. 同时对命名进行调整, 例如0.jpg在映射后变为0000.png  
3. 使用wild6d_pkl_generate.sh将pkl_anntations下整个序列的.pkl文件映射到每一帧单独的_label.pkl文件中. 同时, 为每一帧图像生成单独的_meta.txt文件.
4. 每一帧单独的_label.pkl文件中添加了额外的内容: 相机内参K的fx,fy,cx,cy, 图像的尺寸h,w, 

### pkl_annotations文件的内容
```, '', ''
├──.pkl (dict)
|   ├──'num_frames' (annotations的list长度)
|   ├──'annotations' (list)
|       ├──0 (dict)
|           ├──'name' 'bowl/0049/2021-09-27--12-04-19/0000'
|           ├──'class_id' 1
|           ├──'rotation'  (3, 3)
|           ├──'size'  (3,)
|           ├──'translation'  (3,)
|       ├──1
|       ├──2
|       ├──...
|   ├──'intrinsics'
```

### size与scales的区别
pkl中的size的作用是用于生成nocs空间下的3d bbox，即size存储了点云在nocs中三个维度上的最大值

### 使用该数据集时与NOCS的不同(需要额外处理的部分)

1. 需要单独为每个序列读取内参(因为wild6d每个序列有单独的内参), 存储在_label.pkl文件中
2. (可能的) 单独为每个序列读取图像的尺寸h和w, 存储在_label.pkl文件中


### to do

1. 使用pose_data.py生成_list.txt文件


### CAPTRA代码的修改

1. 读取的_pose.pkl来获取gt位姿, 需要替换为读取_label.pkl文件, key为['poses']