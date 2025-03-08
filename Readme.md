# to do list

- [ ] 目前保存1.png, 2.png, 可选择填充0以及填充位数, 例如0001.png, 0002.png, ...

- [ ] 指定从0开始还是从1开始计数

- [ ] 

# PointCloud Vision Tools
## PointCloudRender

<details>
  <summary>点云可视化函数+参数</summary>
  

PointCloudRender:提供对点云的可视化,输入点云的格式统一为   numpy.array(), shape:(n, 3)

> PointCloudRender(self, result_dir=None, window_shape=(512, 512), window_pos=(50, 25))

初始化

result_dir: path to save captured image 存储图片的路径

pts: width and height of program window 窗口的大小

window_pos: init postion(distance to left and top) of program window 窗口的初始位置

> visualize_shape(self, name, pts, result_dir=None)

简单快速可视化单个点云pts, 使用open3d默认配色方案

name: window name 窗口名

pts: list of pointcloud (numpy.array, nx3) 点云

result_dir: if not None, save image to this path 图片保存路径

> render_multi_pts(self, name, pts_list, color_list, result_dir=None)

同时可视化多个点云, 并可以对不同的点云使用不同配色

示例:

    pcr = PointCloudRender()
    pts1 = ... # (n, 3)
    pts2 = ... # (n, 3)
    color1 = np.array([255, 0, 0])
    color1 = np.array([0, 255, 0])
    render_multi_pts('test', [pts1, pts2], [color1, color2], result_dir=None)

name: windows name 窗口名

pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list

color_list: list of color (every color like np.array([255,0,0])) 存放颜色的list，list长度应该与pts_list相同

result_dir: if result_dir not None, save init img to result_dir 图片保存路径

> render_multi_pts_rotation(self, name, pts_list, color_list, rotate_value=8.0, angle_offset=(0, 0, 0), result_dir=None)

可视化多个点云，进行旋转

pts_list: list of pointcloud (numpy.array, nx3)

color_list: list of color (every color like np.array([255,0,0]))

angle_offset: angle about x,y,z axis, to ajust pointcloud init rotation(every value range -2~2, numpy.float)

result_dir: like: /data/cat, suggest create a folder for single pointcloud
  
</details>


## data preprocessing

data_relabel.sh: 用于将目标文件夹下的图片按顺序排列后重新标记为 0001.png 的格式, 从1开始。 如259.png -> 0259.png, 亦或是根据时间戳保存的图片，只要保存的时间戳是升序排列的，便可以使用该sh来重新标记。

    # 使用实例
    # 推荐使用bash或./ 而非sh(可能会报语法错误), 因为大多数sh指向的是dash, 可以通过ls -la /bin/sh /bin/bash查看sh的具体指向
    bash data_relabel.sh path/to/your/folder/

get_img_list.sh: 用于将目标文件夹下的所有png图像名字保存到文件中, 供其他code使用。

    bash get_img_list.sh path/to/img/folder/ save_file_name
    # 更具体, 如bash get_img_list.sh test/rgb/ img_list.txt
