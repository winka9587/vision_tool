# mask, depth， 以及自定义可视化
# 点云可视化
# 点云投影到2D
# 模型可视化
# 模型投影到2D
import numpy as np
import open3d as o3d
import os

class PointCloudRender:
    def __init__(self, coord, window_shape=(512, 512), window_pos=(50, 25)):
        """
        Args:
            coord: whether show coordinate when visualize pointcloud
            pts: width and height of program window
            window_pos: init postion(distance to left and top) of program window

        """
        self.coord = coord
        self.window_width, self.window_height = window_shape
        self.left, self.top = window_pos
        self.rotate_count = [0]
        self.rotate_result_dir = []

    def visualize_shape(self, name, pts, result_dir=None):
        """ The most simple function, for visualization pointcloud and save image.
        Args:
            name: window name
            pts: list of pointcloud (numpy.array, nx3)
            result_dir: if not None, save image to this path

        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=self.window_width,
                          height=self.window_height, left=self.left, top=self.top)
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        # ctr.rotate(-300.0, 150.0)
        vis.run()
        if result_dir:
            vis.capture_screen_image(os.path.join(result_dir, name + '.png'), False)
        vis.destroy_window()

    def render_multi_pts(self, name, pts_list, color_list, result_dir=None, coord=self.coord):
        """
        Args:
            name: windows name
            pts_list: list of pointcloud (numpy.array, nx3)
            color_list: list of color (every color like np.array([255,0,0]))
            show_img: whether show windows
            result_dir: if result_dir not None, save init img to result_dir
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=self.window_width, height=self.window_height,
                          left=self.left, top=self.top)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = coord
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

        vis.run()
        if result_dir:
            vis.capture_screen_image(os.path.join(result_dir, name + '.png'), False)
        vis.destroy_window()

    def rotate_view(self, vis):
        ctr = vis.get_view_control()
        ctr.set_zoom(2)
        ctr.rotate(8.0, 0)  # 调整值的大小可以调整旋转速度
        if 1 < self.rotate_count[0] < 525:
            save_path = os.path.join(self.rotate_result_dir[0], f'{rotate_count[0]}.png')
            vis.capture_screen_image(save_path, False)
        self.rotate_count[0] += 1
        return False

    def render_multi_pts_rotation(self, name, pts_list, color_list, angle_offset=(0, 0, 0), result_dir=None, coord=self.coord):
        """
            This function is used to render pointcloud in rotation, and save images to result dir
            Attention: this function can't several in the same time,
            if you want do this, check self.rotate_count and rotate_view
        Args:
            pts_list: list of pointcloud (numpy.array, nx3)
            color_list: list of color (every color like np.array([255,0,0]))
            angle_offset: angle about x,y,z axis, to ajust pointcloud init rotation(every value range -2~2, numpy.float)
            result_dir: like: /data/cat, suggest create a folder for single pointcloud
        """
        self.rotate_result_dir.append(result_dir)
        vis = o3d.visualization.Visualizer(result_dir)
        vis.create_window(window_name=name, width=512, height=512, left=50, top=25)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = coord
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
            # define rotation
            # if index == len(pts_list) - 1:
            o3d.visualization.draw_geometries_with_animation_callback([pcd], self.rotate_view)

            pcds.append(pcd)
            vis.add_geometry(pcd)

        o3d.visualization.draw_geometries_with_animation_callback(pcds, self.rotate_view)
        ctr = vis.get_view_control()

        vis.run()
        vis.destroy_window()
        self.rotate_count[0] = 0