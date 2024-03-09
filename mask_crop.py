"""
    input: 
        |---path
            |---color
                |---001.png
                |---002.png
                |---003.png
                |---...
            |---mask
                |---001.png
                |---002.png
                |---003.png
                |---...

    output:
        |---path
             ...
            |---color_mask
                |---001.png
                |---002.png
                |---003.png
                |---...


"""
import cv2
import numpy as np
import os
from tqdm import tqdm  # 引入tqdm库

def apply_mask(color_dir, mask_dir, output_dir, bg_color, value):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取mask目录中所有文件的列表
    mask_files = os.listdir(mask_dir)
    
    # 使用tqdm创建一个进度条
    for mask_file in tqdm(mask_files, desc='Processing'):
        mask_path = os.path.join(mask_dir, mask_file)
        
        # 获取对应的color图像路径
        color_path = os.path.join(color_dir, mask_file)
        
        if os.path.exists(color_path):
            # 读取mask和color图像
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(color_path)
            
            # 创建一个布尔数组，标记mask中值为value的点
            selected_points = mask == value
            
            # 将布尔数组扩展为与color图像相同的形状，用于后续操作
            selected_points_3d = np.repeat(selected_points[:, :, np.newaxis], 3, axis=2)
            
            # 创建一个与color图像大小相同，全为bg_color的图像
            bg = np.full(color.shape, bg_color, dtype=np.uint8)
            
            # 使用selected_points_3d更新bg图像的选中区域
            bg[selected_points_3d] = color[selected_points_3d]
            
            # 保存修改后的图像到输出目录
            output_path = os.path.join(output_dir, mask_file)
            cv2.imwrite(output_path, bg)


if __name__ == "__main__":
    # 设置目录和背景颜色
    # color_dir = '/data4/cxx/dataset/gs/storm2/color/'
    # mask_dir = '/data4/cxx/dataset/gs/storm2/mask/'
    # output_dir = '/data4/cxx/dataset/gs/storm2/color_mask'
    color_dir = '/data4/cxx/dataset/gs/storm1/color/'
    mask_dir = '/data4/cxx/dataset/gs/storm1/mask/'
    output_dir = '/data4/cxx/dataset/gs/storm1/color_mask'
    bg_color = (255, 255, 255)  # 例如，白色背景
    value = 1  # mask中被选中的点的值

    # 应用mask并保存结果
    apply_mask(color_dir, mask_dir, output_dir, bg_color, value)