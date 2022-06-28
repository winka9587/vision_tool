# 根据id选择mask像素点
# 根据bool选择mask像素点
# get_bbox
# 裁剪图像
# 获得choose

# 可视化mask
def viz_mask_bool(name, mask):
    mask_show = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    idx_ = np.where(mask)
    mask_show[idx_[0], idx_[1], :] = 255
    # cv2.imshow('rgb', raw_rgb)
    # cv2.waitKey(0)
    cv2.imshow(name, mask_show)
    cv2.waitKey(0)