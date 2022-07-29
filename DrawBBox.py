# coding=utf-8
import cv2
import numpy as np
from utils import load_obj

def project_PVNet(pts_3d, intrinsic_matrix):
    pts_2d = np.matmul(pts_3d, intrinsic_matrix.T)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d

def viz_target_on_img(target_, intrinsic_, img_, r ,g, b):
    img = img_.copy()
    uv_target_ = project_PVNet(target_, intrinsic_).astype(np.int32)
    uv = []
    for u_, v_ in uv_target_:
        uv.append((u_, v_)) # orgin
        if 0 < v_ < img.shape[0] and 0 < u_ < img.shape[1]:
            pass
            # img[v_, u_, 0] = b
            # img[v_, u_, 1] = g
            # img[v_, u_, 2] = r
    return img, uv

# main()
if __name__ == '__main__':

    color = cv2.imread('M:/jl/1.png')
    obj_points, _ = load_obj('M:/jl/Cat.obj')

    max_x, max_y, max_z = np.max(obj_points, 0)
    min_x, min_y, min_z = np.min(obj_points, 0)
    max_x = max_x.item()
    max_y = max_y.item()
    max_z = max_z.item()
    min_x = min_x.item()
    min_y = min_y.item()
    min_z = min_z.item()
    corners = np.array([
        [max_x, max_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, min_z],

        [max_x, max_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [min_x, min_y, max_z],
    ])
    intrinsics = np.array([[482.9516, 0, 321.5346], [0, 482.7811, 253.04215], [0, 0, 1]])
    my_R = np.array(
        [0.503134, 0.737461, 0.450565,
         0.648271, 0.0227067, -0.76107,
         -0.57149, 0.67501, -0.466648])
    my_R = my_R.reshape(3, 3)
    my_t = np.array([46.6211, 23.0785, 757.451])

    corners_pred = np.dot(corners, my_R.T) + my_t

    img_ = color
    img_, uv_pred = viz_target_on_img(corners_pred, intrinsics, img_, r=0, g=0, b=255)
    thickness = 2
    gt_color = (0, 0, 255)
    # pred_color = testdataset.color_list[obj_]
    pred_color = (253,251,117)
    # pred
    cv2.line(img_, uv_pred[0], uv_pred[1], pred_color, thickness)
    cv2.line(img_, uv_pred[0], uv_pred[2], pred_color, thickness)
    cv2.line(img_, uv_pred[1], uv_pred[3], pred_color, thickness)
    cv2.line(img_, uv_pred[2], uv_pred[3], pred_color, thickness)

    cv2.line(img_, uv_pred[4], uv_pred[5], pred_color, thickness)
    cv2.line(img_, uv_pred[4], uv_pred[6], pred_color, thickness)
    cv2.line(img_, uv_pred[5], uv_pred[7], pred_color, thickness)
    cv2.line(img_, uv_pred[6], uv_pred[7], pred_color, thickness)

    cv2.line(img_, uv_pred[0], uv_pred[4], pred_color, thickness)
    cv2.line(img_, uv_pred[1], uv_pred[5], pred_color, thickness)
    cv2.line(img_, uv_pred[2], uv_pred[6], pred_color, thickness)
    cv2.line(img_, uv_pred[3], uv_pred[7], pred_color, thickness)

    cv2.imshow('project target', img_)
    cv2.waitKey(0)

    save = True
    save_path = "M:/jl/res.png"
    if save:
        cv2.imwrite(save_path, img_)
        print(f"save img to {save_path}")
