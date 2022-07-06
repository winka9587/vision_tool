import os
import cv2
import numpy as np


def pjoin(*a):
    """
        A path join function suitable for both ubuntu and windows
        ������ubuntu��windows��·��ƴ�Ӻ���
    Args:
        *a: The path waitting for join
        ʹ��ʾ��: path=pjoin('/data1/','001.png')
    """
    path = a[0]
    for i in range(len(a)):
        if i==0:
            continue
        else:
            path = os.path.join(path, a[i]).replace('\\', '/')
    return path


def load_obj(path_to_file):
    """
        ����objģ��
    Args:
        path_to_file: objģ��·��
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
        �������ͼ
    Args:
        depth_path: ���ͼ·��
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
        �������ĳ�����bbox����������(x1, y1)��(x2, y2)�� ���������ͼ���ϵ������βü��������ڲü��Լ�֮��ľ���Ȳ���
    Args:
        bbox: ����� (y1, x1, y2, x2)
        img_heigth: ͼ��ĸ�
        img_width: ͼ��Ŀ�
    Return:
        rmin, rmax, cmin, cmax: �ü�ͼ��,����: img[rmin:rmax, cmin:cmax, :]
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
