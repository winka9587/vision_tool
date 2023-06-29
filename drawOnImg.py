
import cv

"""
    This code comes from https://github.com/winka9587/zzh-linemod-viz/blob/f5a4a75a758aa3ba2645cf189bfefc09b1d8b2a4/tools/eval_LM2_occ.py
    used to viz bbox on img about PVNet occ-linemod dataset
"""

def get_corners(model):
    # Variable(t).cuda()
    max_x, max_y, max_z = torch.max(model, 0)[0]
    min_x, min_y, min_z = torch.min(model, 0)[0]
    max_x = max_x.item()
    max_y = max_y.item()
    max_z = max_z.item()
    min_x = min_x.item()
    min_y = min_y.item()
    min_z = min_z.item()
    corners = torch.tensor([
        [max_x, max_y, min_z],
        [max_x, min_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, min_z],

        [max_x, max_y, max_z],
        [max_x, min_y, max_z],
        [min_x, max_y, max_z],
        [min_x, min_y, max_z],
    ])

def project_PVNet(pts_3d, intrinsic_matrix):
    pts_2d = np.matmul(pts_3d, intrinsic_matrix.T)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d
def viz_target_on_img(target_, intrinsic_, img_, r ,g, b):
    img = img_.copy()
    uv_target_ = project_PVNet(target_, intrinsic_).astype(np.int32)
    uv = []
    for u_, v_ in uv_target_:
        uv.append((u_, v_))  # orgin
        if 0 < v_ < img.shape[0] and 0 < u_ < img.shape[1]:
            pass
            # img[v_, u_, 0] = b
            # img[v_, u_, 1] = g
            # img[v_, u_, 2] = r
    return img, uv


"""
    根据mask, 获得2d包围盒(正方形, 可用于卷积等操作)
    input:
        y1, x1, y2, x2 = bbox
        (x1, y1)为左下角坐标
        (x2, y2)为右上角坐标(x轴为水平, y轴数值的坐标系下)
    output:
        rmin, rmax, cmin, cmax : (c: col, r: row)
        
    
    40是根据经验得到的
    min(window_size, 440)限制了包围盒的最大大小, 可根据实际使用进行修改
"""
def get_bbox(bbox, width=640, height=480):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_height = height
    img_width = width
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


"""
    在图像上绘制2d包围盒
    input:
        img_input: (w, h, 3)
        bbox: list, cmin, cmax, rmin, rmax
        (c: col, r: row)
    output:
        img: img with 2d bbox
"""
def draw_bounding_box(img_input, bbox, color=[0, 255, 0], title="Image with Bounding Box"):
    cmin, cmax, rmin, rmax = bbox
    r, g, b = color
    img = img_input.copy()
    # 在图像上绘制包围盒
    cv2.rectangle(img, (cmin, rmin), (cmax, rmax), (r, g, b), 2)

    # 显示带有包围盒的图像
    cv2.imshow(title, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img



def run(save_path="", save=False):
    intrinsics = np.array([[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]])
    img_ = color
    img_, uv_gt = viz_target_on_img(corners_gt, intrinsics, img_, r=0, g=255, b=0)
    img_, uv_pred = viz_target_on_img(corners_pred, intrinsics, img_, r=0, g=0, b=255)
    thickness = 2
    gt_color = (0, 0, 255)
    pred_color = (160, 242, 189)
    # gt
    cv2.line(img_, uv_gt[0], uv_gt[1], gt_color, thickness)
    cv2.line(img_, uv_gt[0], uv_gt[2], gt_color, thickness)
    cv2.line(img_, uv_gt[1], uv_gt[3], gt_color, thickness)
    cv2.line(img_, uv_gt[2], uv_gt[3], gt_color, thickness)

    cv2.line(img_, uv_gt[4], uv_gt[5], gt_color, thickness)
    cv2.line(img_, uv_gt[4], uv_gt[6], gt_color, thickness)
    cv2.line(img_, uv_gt[5], uv_gt[7], gt_color, thickness)
    cv2.line(img_, uv_gt[6], uv_gt[7], gt_color, thickness)

    cv2.line(img_, uv_gt[0], uv_gt[4], gt_color, thickness)
    cv2.line(img_, uv_gt[1], uv_gt[5], gt_color, thickness)
    cv2.line(img_, uv_gt[2], uv_gt[6], gt_color, thickness)
    cv2.line(img_, uv_gt[3], uv_gt[7], gt_color, thickness)
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
    if save:
        cv2.imwrite(save_path, img_)
        print(f"save img to {save_path}")