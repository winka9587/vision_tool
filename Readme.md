# PointCloud Vision Tools
## PointCloudRender

PointCloudRender:�ṩ�Ե��ƵĿ��ӻ�,������Ƶĸ�ʽͳһΪ   numpy.array(), shape:(n, 3)

> PointCloudRender(self, result_dir=None, window_shape=(512, 512), window_pos=(50, 25))

��ʼ��

result_dir: path to save captured image �洢ͼƬ��·��

pts: width and height of program window ���ڵĴ�С

window_pos: init postion(distance to left and top) of program window ���ڵĳ�ʼλ��

> visualize_shape(self, name, pts, result_dir=None)

�򵥿��ٿ��ӻ���������pts, ʹ��open3dĬ����ɫ����

name: window name ������

pts: list of pointcloud (numpy.array, nx3) ����

result_dir: if not None, save image to this path ͼƬ����·��

> render_multi_pts(self, name, pts_list, color_list, result_dir=None)

ͬʱ���ӻ��������, �����ԶԲ�ͬ�ĵ���ʹ�ò�ͬ��ɫ

ʾ��:

    pcr = PointCloudRender()
    pts1 = ... # (n, 3)
    pts2 = ... # (n, 3)
    color1 = np.array([255, 0, 0])
    color1 = np.array([0, 255, 0])
    render_multi_pts('test', [pts1, pts2], [color1, color2], result_dir=None)

name: windows name ������

pts_list: list of pointcloud (numpy.array, nx3) ��ŵ��Ƶ�list

color_list: list of color (every color like np.array([255,0,0])) �����ɫ��list��list����Ӧ����pts_list��ͬ

result_dir: if result_dir not None, save init img to result_dir ͼƬ����·��

> render_multi_pts_rotation(self, name, pts_list, color_list, rotate_value=8.0, angle_offset=(0, 0, 0), result_dir=None)

���ӻ�������ƣ�������ת

pts_list: list of pointcloud (numpy.array, nx3)

color_list: list of color (every color like np.array([255,0,0]))

angle_offset: angle about x,y,z axis, to ajust pointcloud init rotation(every value range -2~2, numpy.float)

result_dir: like: /data/cat, suggest create a folder for single pointcloud