# import trimesh
import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist

# def load_obj_as_pointcloud(obj_path, sample_points=-1, random_seed=0):
#     # 使用 trimesh 加载 obj 文件
#     mesh = trimesh.load(obj_path, process=False)
#     print("sample num: {}".format(sample_points))

#     # 采样点云
#     if sample_points == -1:
#         # 如果 sample_points 为 -1，使用模型的所有顶点作为点云
#         points = mesh.vertices
#     else:
#         # 否则，使用 trimesh.sample.sample_surface 进行采样
#         points, _ = trimesh.sample.sample_surface(mesh, sample_points)
#     print("load {} \n shape: {}".format(obj_path, points.shape))
#     return points

def sample_vertices(vertices, num_samples, random_seed=0):
    """
    从顶点数据中随机采样一定数量的点。

    :param vertices: np.array, 顶点数据，形状为 (N, 3)
    :param num_samples: int, 需要采样的点的数量
    :param random_seed: int, 随机种子以确保结果可重复
    :return: np.array, 采样后的点云数据，形状为 (num_samples, 3)
    """
    np.random.seed(random_seed)
    
    if num_samples > len(vertices):
        raise ValueError("采样数量大于可用顶点数量")
    
    sampled_indices = np.random.choice(len(vertices), num_samples, replace=False)
    sampled_vertices = vertices[sampled_indices]
    
    return sampled_vertices

def load_obj_vertices_only(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # 仅提取顶点信息
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
    return np.array(vertices)

def load_obj_as_pointcloud(obj_path, sample_points=-1):
    # 使用 Open3D 加载 obj 文件
    pointcloud = load_obj_vertices_only(obj_path)

    # 获取顶点数
    num_vertices = pointcloud.shape[0]
    # 如果需要，可以对点云进行采样
    if sample_points == -1:
        pass
    else:
        # 使用 Open3D 的采样方法
        pointcloud = sample_vertices(pointcloud, sample_points)

    print("load {} \n n:{} \n shape: {}".format(obj_path, num_vertices, pointcloud.shape))
    return pointcloud

# def load_obj_as_pointcloud(obj_path, sample_points=-1):
#     # 使用 Open3D 加载 obj 文件
#     mesh = o3d.io.read_triangle_mesh(obj_path)

#     # 获取顶点数
#     num_vertices = np.asarray(mesh.vertices).shape[0]
#     # # 如果需要，可以对点云进行采样
#     # if sample_points == -1:
#     #     # 不进行采样，使用所有顶点
#     #     pointcloud = np.asarray(mesh.vertices)
#     # else:
#     #     # 使用 Open3D 的采样方法
#     #     pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
#     #     pointcloud = np.asarray(pcd.points)
#     pointcloud = np.asarray(mesh.vertices)
#     print("load {} \n n:{} \n shape: {}".format(obj_path, num_vertices, pointcloud.shape))
#     return pointcloud

def compute_hausdorff_statistics(pointcloud1, pointcloud2):
    # 计算从 pointcloud1 到 pointcloud2 的最近点距离
    dists_1_to_2 = cdist(pointcloud1, pointcloud2).min(axis=1)
    
    # 计算从 pointcloud2 到 pointcloud1 的最近点距离
    dists_2_to_1 = cdist(pointcloud2, pointcloud1).min(axis=1)
    
    # Hausdorff 距离是两个方向的最大值
    hausdorff_dist = max(dists_1_to_2.max(), dists_2_to_1.max())

    # 计算统计信息
    min_dist = dists_1_to_2.min()
    max_dist = dists_1_to_2.max()
    mean_dist = dists_1_to_2.mean()
    rms_dist = np.sqrt(np.mean(dists_1_to_2**2))
    
    return hausdorff_dist, min_dist, max_dist, mean_dist, rms_dist

# 示例：加载两个模型的路径
# obj_path1 = 'D:/RBOT-results/RBOT-results/rbot-models/ape.obj'
# obj_path2 = 'D:/RBOT-results/RBOT-results/3d3r6_angle3/ape/model1.obj'
obj_path2 = 'D:/RBOT-results/RBOT-results/rbot-models/1111.obj'
obj_path1 = 'D:/RBOT-results/RBOT-results/rbot-models/2222.obj'


# 加载模型并转换为点云
# pointcloud1 = load_obj_as_pointcloud(obj_path1, 1302)
pointcloud1 = load_obj_as_pointcloud(obj_path1)
pointcloud2 = load_obj_as_pointcloud(obj_path2)

# 计算Hausdorff距离和统计信息
hausdorff_dist, min_dist, max_dist, mean_dist, rms_dist = compute_hausdorff_statistics(pointcloud1, pointcloud2)

# 计算 BBox Diag
bbox_diag = np.linalg.norm(pointcloud1.ptp(axis=0))  # 使用点云1的对角线
min_dist_bbox = min_dist / bbox_diag
max_dist_bbox = max_dist / bbox_diag
mean_dist_bbox = mean_dist / bbox_diag
rms_dist_bbox = rms_dist / bbox_diag

# 输出结果
print(f"Sampled {len(pointcloud1)} pts on model1 searched closest on model2")
print(f"min : {min_dist:.6f} max : {max_dist:.6f} mean : {mean_dist:.6f} RMS : {rms_dist:.6f}")
print(f"Values w.r.t. BBox Diag ({bbox_diag:.6f})")
print(f"min : {min_dist_bbox:.6f} max : {max_dist_bbox:.6f} mean : {mean_dist_bbox:.6f} RMS : {rms_dist_bbox:.6f}")

