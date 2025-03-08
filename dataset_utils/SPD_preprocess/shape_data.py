import os
import sys
import h5py
import glob
import numpy as np
import _pickle as cPickle
from lib.utils import sample_points_from_mesh


def save_nocs_model_to_file(obj_model_dir):
    """ Sampling points from mesh model and normalize to NOCS.
        Models are centered at origin, i.e. NOCS-0.5

    """
    mug_meta = {}
    # used for re-align mug category
    special_cases = {'3a7439cfaa9af51faf1af397e14a566d': np.array([0.115, 0.0, 0.0]),
                     '5b0c679eb8a2156c4314179664d18101': np.array([0.083, 0.0, -0.044]),
                     '649a51c711dc7f3b32e150233fdd42e9': np.array([0.0, 0.0, -0.017]),
                     'bf2b5e941b43d030138af902bc222a59': np.array([0.0534, 0.0, 0.0]),
                     'ca198dc3f7dc0cacec6338171298c66b': np.array([0.120, 0.0, 0.0]),
                     'f42a9784d165ad2f5e723252788c3d6e': np.array([0.117, 0.0, -0.026])}

    # # CAMERA dataset
    for subset in ['train', 'val']:
        print("CAMERA-{}".format(subset))
        camera = {}
        for synsetId in ['02876657', '02880940', '02942699', '02946921', '03642806', '03797390']:
            synset_dir = os.path.join(obj_model_dir, subset, synsetId)
            inst_list = sorted(os.listdir(synset_dir))
            for instance in inst_list:
                path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                # 对模型进行采样
                model_points = sample_points_from_mesh(path_to_mesh_model, 1024, fps=True, ratio=3)
                # flip z-axis in CAMERA   # 反转Z轴
                model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                # re-align mug category
                # 对一部分模型加偏移量(上面的字典)来重新对齐
                if synsetId == '03797390':
                    if instance == 'b9be7cfe653740eb7633a2dd89cec754':
                        # skip this instance in train set, improper mug model, only influence training.
                        continue
                    if instance in special_cases.keys():
                        shift = special_cases[instance]
                    else:
                        shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
                        shift = np.array([shift_x, 0.0, 0.0])
                    model_points += shift
                    size = 2 * np.amax(np.abs(model_points), axis=0)
                    scale = 1 / np.linalg.norm(size)
                    model_points *= scale
                    mug_meta[instance] = [shift, scale]
                camera[instance] = model_points
        with open(os.path.join(obj_model_dir, 'camera_{}.pkl'.format(subset)), 'wb') as f:
            # 将camera对象保存到文件f(obj_model_dir/camera_train.pkl和camera_val.pkl)中
            cPickle.dump(camera, f)
    # Real dataset
    # 真实数据集中，例如real_train目录下，每个模型对应.obj,norm.txt,norm_vertices.txt三个文件
    for subset in ['real_train', 'real_test']:
        print("REAL-{}".format(subset))
        real = {}
        inst_list = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in inst_list:
            # os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值
            # glob返回的是路径，所以要用basename(去目录)和split(去类型后缀)来截取obj的文件名,最后得到instance
            instance = os.path.basename(inst_path).split('.')[0]
            bbox_file = inst_path.replace('.obj', '.txt')  # inst_path是路径/文件名.obj
            bbox_dims = np.loadtxt(bbox_file)
            scale = np.linalg.norm(bbox_dims)  # 默认计算各元素平方和再开根号
            model_points = sample_points_from_mesh(inst_path, 1024, fps=True, ratio=3)
            model_points /= scale
            # relable mug category
            # 如果是马克杯的文件，还要多进行额外的处理
            if 'mug' in instance:
                # z轴的最小值减去最大值除以2，再减去x的最小值？
                # 相当于z轴长度的一半的负值，减去x的最小值(负值取负)
                shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
                shift = np.array([shift_x, 0.0, 0.0])
                model_points += shift
                # 所有点的xyz坐标都取绝对值,size选择距离中心最远的点，到中心点的距离乘以2
                # size和scale的概念有什么区别
                size = 2 * np.amax(np.abs(model_points), axis=0)
                scale = 1 / np.linalg.norm(size)
                model_points *= scale
                mug_meta[instance] = [shift, scale]
            real[instance] = model_points
        # 将处理后的数据(从obj中读到的点，与文件名对应起来生成的一个字典real存出到pkl文件中
        with open(os.path.join(obj_model_dir, '{}.pkl'.format(subset)), 'wb') as f:
            cPickle.dump(real, f)
    # save mug_meta information for re-labeling
    # 为什么给mug单独建了一个pkl？
    with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'wb') as f:
        cPickle.dump(mug_meta, f)


# n_points是每个模型采样多少个点
def save_model_to_hdf5(obj_model_dir, n_points, fps=False, include_distractors=False, with_normal=False):
    """ Save object models (point cloud) to HDF5 file.
        Dataset used to train the auto-encoder.
        Only use models from ShapeNetCore.
        Background objects are not inlcuded as default. We did not observe that it helps
        to train the auto-encoder.

    """
    # 我理解的是，obj_model是来自ShapeNetCore数据集的模型，只挑选了NOCS中的6个类别的模型来进行训练
    # 可以可视化看看是不是
    catId_to_synsetId = {1: '02876657', 2: '02880940', 3: '02942699', 4: '02946921', 5: '03642806', 6: '03797390'}
    distractors_synsetId = ['00000000', '02954340', '02992529', '03211117']
    with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)
    # read all the paths to models
    print('Sampling points from mesh model ...')
    # with_normal的作用就是train_data和val_data生成维度是 m*n*6 还是 m*n*3
    # n_points有4096(不使用FPS)和2048(使用FPS，计算mean shape)两种
    # 这里的3000和500都是取了一个比较大的数，在最后会进行切片
    if with_normal:
        train_data = np.zeros((3000, n_points, 6), dtype=np.float32)
        val_data = np.zeros((500, n_points, 6), dtype=np.float32)
    else:
        train_data = np.zeros((3000, n_points, 3), dtype=np.float32)
        val_data = np.zeros((500, n_points, 3), dtype=np.float32)
    train_label = []
    val_label = []
    train_count = 0
    val_count = 0
    # CAMERA
    for subset in ['train', 'val']:
        for catId in range(1, 7):
            synset_dir = os.path.join(obj_model_dir, subset, catId_to_synsetId[catId])
            # 将一个类别下的所有实例都拿出来
            inst_list = sorted(os.listdir(synset_dir))
            for instance in inst_list:
                # 一个instance是一个文件名，其下包含model.obj，bbox.txt和model.mtl
                path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                if instance == 'b9be7cfe653740eb7633a2dd89cec754':
                    continue
                model_points = sample_points_from_mesh(path_to_mesh_model, n_points, with_normal, fps=fps, ratio=2)
                model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                # 看看类别6是不是mug
                if catId == 6:
                    shift = mug_meta[instance][0]
                    scale = mug_meta[instance][1]
                    model_points = scale * (model_points + shift)
                if subset == 'train':
                    train_data[train_count] = model_points  # 点云
                    train_label.append(catId)  # 类标签
                    train_count += 1
                else:
                    # val
                    val_data[val_count] = model_points
                    val_label.append(catId)
                    val_count += 1
                # 最终生成了train_data和train_label，val_data和val_label
                # xxx_data和xxx_label分别是点云+该点云对应的类标签
        # distractors
        # include_distractors参数来决定是否向前面的数据集中添加干扰
        # 添加的干扰项类标签为0
        if include_distractors:
            for synsetId in distractors_synsetId:
                synset_dir = os.path.join(obj_model_dir, subset, synsetId)
                inst_list = sorted(os.listdir(synset_dir))
                for instance in inst_list:
                    path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                    model_points = sample_points_from_mesh(path_to_mesh_model, n_points, with_normal, fps=fps, ratio=2)
                    # TODO: check whether need to flip z-axis, currently not used
                    model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                    if subset == 'train':
                        train_data[train_count] = model_points
                        train_label.append(0)
                        train_count += 1
                    else:
                        val_data[val_count] = model_points
                        val_label.append(0)
                        val_count += 1
    # Real
    # 这段代码主要就是因为syn和real数据集的结构不一样，作用都是相同的
    # 虽然相比于syn多读取了bbox.txt，但那也是因为real数据集的模型没有归一化，读取txt只是为了获得scale
    # （对比一下syn的bbox看看）
    for subset in ['real_train', 'real_test']:
        path_to_mesh_models = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in sorted(path_to_mesh_models):
            instance = os.path.basename(inst_path).split('.')[0]
            if instance.startswith('bottle'):
                catId = 1
            elif instance.startswith('bowl'):
                catId = 2
            elif instance.startswith('camera'):
                catId = 3
            elif instance.startswith('can'):
                catId = 4
            elif instance.startswith('laptop'):
                catId = 5
            elif instance.startswith('mug'):
                catId = 6
            else:
                raise NotImplementedError
            model_points = sample_points_from_mesh(inst_path, n_points, with_normal, fps=fps, ratio=2)
            bbox_file = inst_path.replace('.obj', '.txt')
            bbox_dims = np.loadtxt(bbox_file)
            model_points /= np.linalg.norm(bbox_dims)
            if catId == 6:
                shift = mug_meta[instance][0]
                scale = mug_meta[instance][1]
                model_points = scale * (model_points + shift)
            if subset == 'real_train':
                train_data[train_count] = model_points
                train_label.append(catId)
                train_count += 1
            else:
                val_data[val_count] = model_points
                val_label.append(catId)
                val_count += 1

    num_train_instances = len(train_label)
    num_val_instances = len(val_label)
    assert num_train_instances == train_count
    assert num_val_instances == val_count
    # 进行切片，扔到多余部分
    train_data = train_data[:num_train_instances]
    val_data = val_data[:num_val_instances]
    train_label = np.array(train_label, dtype=np.uint8)
    val_label = np.array(val_label, dtype=np.uint8)
    print('{} shapes found in train dataset'.format(num_train_instances))
    print('{} shapes found in val dataset'.format(num_val_instances))

    # write to HDF5 file
    print('Writing data to HDF5 file ...')
    if with_normal:
        filename = 'ShapeNetCore_{}_with_normal.h5'.format(n_points)
    else:
        filename = 'ShapeNetCore_{}.h5'.format(n_points)
    hfile = h5py.File(os.path.join(obj_model_dir, filename), 'w')
    train_dataset = hfile.create_group('train')
    train_dataset.attrs.create('len', num_train_instances)
    train_dataset.create_dataset('data', data=train_data, compression='gzip', dtype='float32')
    train_dataset.create_dataset('label', data=train_label, compression='gzip', dtype='uint8')
    val_dataset = hfile.create_group('val')
    val_dataset.attrs.create('len', num_val_instances)
    val_dataset.create_dataset('data', data=val_data, compression='gzip', dtype='float32')
    val_dataset.create_dataset('label', data=val_label, compression='gzip', dtype='uint8')
    hfile.close()


if __name__ == '__main__':
    obj_model_dir = '/data2/cxx/dataset/NOCS/obj_models'
    # Save ground truth models for training deform network
    # 将CAMERA数据集和Real数据集的模型处理成字典key(模型名):value(模型对应的points:1024x3大小)存出到pkl文件中
    print('save_nocs_model_to_file... start')
    save_nocs_model_to_file(obj_model_dir)
    # 两次调用save_model_to_hdf5文件，不同之处在于n_points不同，以及是否采用FPS
    # Save models to HDF5 file for training the auto-encoder.
    # 不采用FPS，采样n_points=4096 = 2048x2
    print('save model to hdf5 4096... start')
    save_model_to_hdf5(obj_model_dir, n_points=4096, fps=False)
    # Save nmodels to HDF5 file, which used to generate mean shape.
    # 这些模型通过FPS采样来(n_points更少，比之前少了一倍)来生成mean shape，也就是类平均模型
    print('save model to hdf5 2048... start')
    save_model_to_hdf5(obj_model_dir, n_points=2048, fps=True)

    # import random
    # import open3d as o3d
    # for file in ['camera_train.pkl', 'camera_val.pkl', 'real_train.pkl', 'real_test.pkl']:
    #     with open(os.path.join(obj_model_dir, file), 'rb') as f:
    #         obj_models = cPickle.load(f)
    #     instance = random.choice(list(obj_models.keys()))
    #     model_points = obj_models[instance]
    #     print('Diameter: {}'.format(np.linalg.norm(2*np.amax(np.abs(model_points), axis=0))))
    #     color = np.repeat(np.array([[1, 0, 0]]), model_points.shape[0], axis=0)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(model_points)
    #     pcd.colors = o3d.utility.Vector3dVector(color)
    #     # visualization: camera coordinate frame
    #     points = [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    #     lines = [[0, 1], [0, 2], [0, 3]]
    #     colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(points)
    #     line_set.lines = o3d.utility.Vector2iVector(lines)
    #     line_set.colors = o3d.utility.Vector3dVector(colors)
    #     o3d.visualization.draw_geometries([pcd, line_set])
