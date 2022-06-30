# coding=utf-8
# mask, depth， 以及自定义可视化
# 点云可视化
# 点云投影到2D
# 模型可视化
# 模型投影到2D
import time
import numpy as np
import open3d as o3d
import os
import cv2
from utils import pjoin


class PointCloudRender:
    def __init__(self, result_dir=None, window_shape=(512, 512), window_pos=(50, 25)):
        """
        Args:
            result_dir: path to save captured image 存储图片的路径
            pts: width and height of program window 窗口的大小
            window_pos: init postion(distance to left and top) of program window 窗口的初始位置
        """
        self.window_width, self.window_height = window_shape
        self.left, self.top = window_pos
        self.rotate_count = [0]
        self.rotate_value = 8.0
        self.result_dir = result_dir

    def ____rotate_view(self, vis):
        ctr = vis.get_view_control()
        # ctr.set_zoom(2)
        ctr.rotate(self.rotate_value, 0)  # 调整值的大小可以调整旋转速度
        if self.result_dir and 1 < self.rotate_count[0] < 525:
            save_path = os.path.join(self.result_dir, f'{self.rotate_count[0]}.png')
            vis.capture_screen_image(save_path, False)
        self.rotate_count[0] += 1
        return False

    def __capture_image(self, vis):
        img_save_path = pjoin(self.result_dir, "{0}.png".format(time.asctime().replace(' ', '-').replace(':', '-')))
        image_buffer = vis.capture_screen_float_buffer()
        img_tmp = np.asarray(image_buffer)
        img = img_tmp.copy()
        img[:, :, 0] = img_tmp[:, :, 2]
        img[:, :, 2] = img_tmp[:, :, 0]

        cv2.imshow("captured image", img)
        cv2.waitKey(1)
        if self.result_dir:

            cv2.imwrite(img_save_path, img * 255.0)
            print("[OK] img save to {0}".format(img_save_path))
        return False

    def visualize_shape(self, name, pts, result_dir=None):
        """ The most simple function, for visualization pointcloud and save image.
        Args:
            name: window name 窗口名
            pts: list of pointcloud (numpy.array, nx3) 点云
            result_dir: if not None, save image to this path 图片保存路径

        """
        self.result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()

        key_to_callback = {}
        key_to_callback[ord("C")] = self.__capture_image
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, window_name=name,
                                                             width=self.window_width, height=self.window_height,
                                                             left=self.left, top=self.top)

    def render_multi_pts(self, name, pts_list, color_list, result_dir=None):
        """
        Args:
            name: windows name 窗口名
            pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list
            color_list: list of color (every color like np.array([255,0,0])) 存放颜色的list，list长度应该与pts_list相同
            result_dir: if result_dir not None, save init img to result_dir 图片保存路径
        """
        self.result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        assert len(pts_list) == len(color_list)
        pcds = []
        for index in range(len(pts_list)):
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if np.any(color > 1.0):
                color = color.astype(float)/255.0
            colors = np.tile(color, (pts.shape[0], 1))  # reshape color to (pts.shape[0], 1)
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcds.append(pcd)
            vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        key_to_callback = {}
        key_to_callback[ord("C")] = self.__capture_image
        o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback, window_name=name,
                                                             width=self.window_width, height=self.window_height,
                                                             left=self.left, top=self.top)

    def render_multi_pts_rotation(self, name, pts_list, color_list,
                                  rotate_value=8.0, angle_offset=(0, 0, 0), result_dir=None):
        """
            This function is used to render pointcloud in rotation, and save images to result dir
            Attention: this function can't several in the same time,
            if you want do this, check self.rotate_count and __rotate_view
        Args:
            pts_list: list of pointcloud (numpy.array, nx3)
            color_list: list of color (every color like np.array([255,0,0]))
            rotate_value: this value control rotation speed
            angle_offset: angle about x,y,z axis, to ajust pointcloud init rotation(every value range -2~2, numpy.float)
            result_dir: like: /data/cat, suggest create a folder for single pointcloud
        """
        self.rotate_value = rotate_value
        self.result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        assert len(pts_list) == len(color_list)

        pcds = []
        for index in range(len(pts_list)):
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if len(color.shape) == 1 or color.shape[0] != pts.shape[0]:
                if np.any(color > 1.0):
                    color = color.astype(float) / 255
                color = np.tile(color, (pts.shape[0], 1))  # 将color扩大为 (pts.shape[0], 1)
            else:
                print("Error in viz_pts_with_rotation: color shape {0} wrong".format(color.shape))
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(color)
            # 调整pts角度
            x_r, y_r, z_r = angle_offset
            R = pcd.get_rotation_matrix_from_xyz((x_r * np.pi, y_r, z_r * np.pi))
            pcd = pcd.rotate(R, center=(0, 0, 0))

            pcds.append(pcd)
            vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        o3d.visualization.draw_geometries_with_animation_callback(pcds, self.__rotate_view, window_name=name,
                                                             width=self.window_width, height=self.window_height,
                                                             left=self.left, top=self.top)
        self.rotate_count[0] = 0


if __name__ == '__main__':
    # example of usage
    pts1 = np.random.randn(1024, 3)
    pts2 = np.random.randn(1024, 3)
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])

    pcr = PointCloudRender()
    pcr.render_multi_pts("multi", [pts1, pts2], [red, green], "M:/test/")
    pcr.visualize_shape("simple", pts1, "M:/test/")
    pcr.render_multi_pts_rotation("multi-rotate", [pts1, pts2], [red, green])