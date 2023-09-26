# coding=utf-8
import os
import cv2
import numpy as np


def load_obj(path_to_file):
    """
    加载obj模型
    :param path_to_file: obj模型路径
    :return: vertices, faces 顶点 和 面片
    """
    vertices = []
    faces = []
    with open(path_to_file, 'r') as f:
        for line in f:
            if line[:2] == 'v ':
                vertex = line[2:].strip().split(' ')
                vertex = [float(xyz) for xyz in vertex]
                vertices.append(vertex)
            elif line[0] == 'f':
                face = line[1:].replace('//', '/').strip().split(' ')
                face = [int(idx.split('/')[0])-1 for idx in face]
                faces.append(face)
            else:
                continue
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    return vertices, faces


def load_depth(depth_path):
    """
    加载深度图
    :param depth_path: 深度图路径
    :return: depth16 深度图 np.uint16
    """
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def get_bbox(bbox, img_height=480, img_width=640):
    """
    给定检测的长方形bbox的两个坐标(x1, y1)和(x2, y2)， 计算出其在图像上的正方形裁剪区域，用于裁剪以及之后的卷积等操作
    :param bbox: 检测算法的结果 (y1, x1, y2, x2)
    :param img_height: 图像的高
    :param img_width: 图像的宽
    :return: rmin, rmax, cmin, cmax 边界值, 用于裁剪图像,例如: img[rmin:rmax, cmin:cmax, :]
    """
    y1, x1, y2, x2 = bbox
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_height:
        delt = rmax - img_height
        rmax = img_height
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax


def depth2show(depth, norm_type='max'):
    print("norm_type: {}".format(norm_type))
    print("depth max: {}".format(depth.max()))
    if norm_type == 'max':
        show_depth = (depth / depth.max() * 256).astype("uint8")
    else:
        show_depth = (depth / float(norm_type) * 256).astype("uint8")
    return show_depth


def show_depth(depth_path, save_path=None, norm_type='max'):
    depth = load_depth(depth_path)
    depth = depth.astype(np.uint16)
    print("load depth img from {}".format(depth_path))
    depth = depth2show(depth, norm_type)
    cv2.imshow("depth2show", depth)
    cv2.waitKey(0)
    if save_path is not None:
        cv2.imwrite(save_path, depth)
        print("save depth2show img to {}".format(save_path))


def load_txt(txt_path):
    txt_data = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            txt_data.append(data)
    return txt_data


def save_pcd_to_obj(img_path, target_model, sampling_npts=2048, save_path=None):
    # load depth
    depth_path = img_path + '_depth.png'
    depth = load_depth(depth_path)
    # 反投影
    rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
    rgb = rgb[:, :, ::-1]
    coord = cv2.imread(img_path + '_coord.png')[:, :, :3]
    coord = coord[:, :, (2, 1, 0)]
    coord = np.array(coord, dtype=np.float32) / 255
    coord[:, :, 2] = 1 - coord[:, :, 2]
    k_size = 3
    distance_threshold = 2000
    difference_threshold = 10  # 单位mm, 周围的点深度距离超过10mm则不考虑
    point_into_surface = False

    with open(img_path + '_label.pkl', 'rb') as f:
        import _pickle as cPickle
        gts = cPickle.load(f)

    REAL_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
    cam_fx, cam_fy, cam_cx, cam_cy = REAL_intrinsics

    meta_data = load_txt(img_path + '_meta.txt')
    inst_id = None
    for i_ in range(len(meta_data)):
        meta_ = meta_data[i_]
        if meta_[-1] == target_model:
            inst_id = int(meta_[0])
            idx = i_
            break
    if inst_id is None:
        print("Match failed: {}".format(target_model))
        print("meta data: \n{}".format(meta_data))
        return

    # 可视化mask, 确保物体正确
    # bbox_id = gts['instance_ids'][inst_id]
    rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
    mask_raw = cv2.imread(img_path + '_mask.png')[:, :, 2]
    mask = np.equal(mask_raw, inst_id)
    mask = np.logical_and(mask, depth > 0)

    ind = np.where(mask)
    mask2 = np.zeros_like(mask_raw)
    mask2[ind] = 255
    cv2.imshow("mask", mask2)
    cv2.waitKey(0)

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

    if len(choose) > sampling_npts:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:sampling_npts] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, sampling_npts - len(choose)), 'wrap')
    img_width = mask.shape[1]
    img_height = mask.shape[0]
    xmap = np.array([[i_ for i_ in range(640)] for j_ in range(480)])
    ymap = np.array([[j_ for i_ in range(640)] for j_ in range(480)])

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    norm_scale = 1000.0
    pt2 = depth_masked / norm_scale
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    pts = np.concatenate((pt0, pt1, pt2), axis=1)

    # get point
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 可视化该点云
    from PointCloudRender import PointCloudRender
    pcr = PointCloudRender()
    pcr.visualize_shape("simple", pts)
    pcr.render_multi_pts("pcd", [pts], [np.array([0, 0, 255])])

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pcd)
        print("save point cloud to {}".format(save_path))


if __name__ == '__main__':
    # 读取深度图并将可视化的深度图保存
    # in_path = r"F:\BaiduSyncdisk\VR2024\fig\framework\scene6\0066_depth.png"
    # out_path = r"F:\BaiduSyncdisk\VR2024\fig\framework\scene6\0066_depth2show.png"
    # show_depth(in_path, out_path, norm_type='max')

    # # 保存数据中的点云
    # path = r"F:\BaiduSyncdisk\VR2024\fig\framework\0054"
    # target_model = "mug_brown_starbucks_norm"
    # save_path = path + "_2048.ply"
    # save_pcd_to_obj(path, target_model, 2048, save_path)

    # 保存scene2-0007数据中的点云
    # path = r"F:\BaiduSyncdisk\VR2024\fig\teaser\0007"
    # target_model = "mug_daniel_norm"
    # save_path = path + "_2048.ply"
    # save_pcd_to_obj(path, target_model, 2048, save_path)

    # 读取深度图并将可视化的深度图保存
    in_path = r"F:\BaiduSyncdisk\VR2024\fig\teaser\0007_depth.png"
    out_path = r"F:\BaiduSyncdisk\VR2024\fig\teaser\0007_depth2show.png"
    show_depth(in_path, out_path, norm_type='max')