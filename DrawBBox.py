
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
intrinsics = np.array([[572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]])
img_ = color
img_, uv_gt = viz_target_on_img(corners_gt, intrinsics, img_, r=0, g=255, b=0)
img_, uv_pred = viz_target_on_img(corners_pred, intrinsics, img_, r=0, g=0, b=255)
thickness = 2
gt_color = (0, 0, 255)
# pred_color = testdataset.color_list[obj_]
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
