"""
    input:
        pose: 4x4 matrix
        color: numpy.ndarray (w, h, 3)
        depth: numpy.nedarray(w, h, 3), 但三个通道的值是一样的, 取其中一个
        mask: numpy.ndarray(w, h), 其中value为1的像素点为选中的点
        fx,cx,fy,cy: float
        depth_scale: float
    output:
        img_with_3dbbox

    将观测点云通过pose逆变换到物体坐标系下, 然后在物体坐标系下的3个轴上分别取绝对值最大值, 乘以2后作为包围盒的size
    再使用pose将该包围盒的八个点投影到图像上, 并连接对应的点绘制3d包围盒
    
"""
import numpy as np
import cv2
import os

def generate_3dbbox(pose, color, depth, mask, fx, cx, fy, cy, depth_scale, size=None):
    # Convert depth to single channel
    depth = depth[:, :, 0]
    
    # Inverse transform the point cloud to object coordinate system
    inv_pose = np.linalg.inv(pose)
    # object_points = np.matmul(inv_pose[:3, :3], depth.T).T + inv_pose[:3, 3]
    
    # # Calculate the size of the bounding box if not provided
    # if size is None:
    #     size = np.max(np.abs(object_points), axis=0) * 2
    
    # Project the bounding box points onto the image
    bbox_points = np.array([
        [size[0]/2, size[1]/2, size[2]/2],
        [-size[0]/2, size[1]/2, size[2]/2],
        [-size[0]/2, -size[1]/2, size[2]/2],
        [size[0]/2, -size[1]/2, size[2]/2],
        [size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, size[1]/2, -size[2]/2],
        [-size[0]/2, -size[1]/2, -size[2]/2],
        [size[0]/2, -size[1]/2, -size[2]/2]
    ])
    bbox_points = np.matmul(pose[:3, :3], bbox_points.T).T + pose[:3, 3]
    bbox_points = np.matmul(np.array([[fx, 0, cx], [0, fy, cy]]), bbox_points.T).T
    bbox_points = bbox_points[:, :2] / bbox_points[:, 2:]
    
    # Draw the 3D bounding box on the image
    img_with_3dbbox = color.copy()
    for i in range(4):
        cv2.line(img_with_3dbbox, tuple(bbox_points[i]), tuple(bbox_points[(i+1)%4]), (0, 255, 0), 2)
        cv2.line(img_with_3dbbox, tuple(bbox_points[i+4]), tuple(bbox_points[((i+1)%4)+4]), (0, 255, 0), 2)
        cv2.line(img_with_3dbbox, tuple(bbox_points[i]), tuple(bbox_points[i+4]), (0, 255, 0), 2)
    
    # Return the img_with_3dbbox
    return img_with_3dbbox


"""
K: p[642.182 361.587]  f[644.525 644.525]

[0.58958673, 0.47375342, 0.6541752],
[[ 0.11325313,  0.00365644, -0.1530459,   0.03787684],
 [ 0.14508761, -0.06330656,  0.10585156, -0.03060192],
 [-0.04884674, -0.17955945, -0.04043619,  0.660544  ],
 [ 0.,          0.,          0.,          1.        ]],

[0.6230198,  0.43719524, 0.64861906],
[[ 0.110953,    0.08763637,  0.16121694, -0.00240512],
 [-0.14579692, -0.07229181,  0.13963792, -0.02877988],
 [ 0.11141941, -0.18186633,  0.02217996,  0.583863  ],
 [ 0.,          0.,          0.,          1.        ]],

[0.6444633,  0.41034156, 0.64520293],
[[ 0.11073474,  0.15148403,  0.13977996, -0.02876984],
 [-0.13731359, -0.06412298,  0.17827296, -0.04313501],
 [ 0.15372321, -0.16639972,  0.058552,    0.6500201 ],
 [ 0.,          0.,          0.,          1.        ]],

[0.59172165, 0.473758,   0.6522414],
[[ 0.11489279,  0.01822667, -0.14449887,  0.00279258],
 [ 0.1327638,  -0.08877916,  0.09436376, -0.06700505],
 [-0.05988243, -0.16185962, -0.06802974,  0.707853  ],
 [ 0.,          0.,          0.,          1.        ]]

"""
if __name__ == "__main__":
    # Test the function
    size = np.array([0.58958673, 0.47375342, 0.6541752])
    pose = np.array([
        [ 0.11325313,  0.00365644, -0.1530459,   0.03787684],
        [ 0.14508761, -0.06330656,  0.10585156, -0.03060192],
        [-0.04884674, -0.17955945, -0.04043619,  0.660544  ],
        [ 0.,          0.,          0.,          1.        ]
        ])
    image_path = "/data4/cxx/dataset/gs/mug1/"
    file_name = "00000.png"
    color_path = os.path.join(image_path, 'color', file_name)
    depth_path = os.path.join(image_path, 'depth', file_name)
    mask_path = os.path.join(image_path, 'mask', file_name)
    fx, fy, cx, cy = 644.525, 644.525, 642.182, 361.587
    depth_scale = 0.001
    color = cv2.imread(color_path)
    depth = cv2.imread(depth_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img_with_3dbbox = generate_3dbbox(pose, color, depth, mask, fx, cx, fy, cy, depth_scale, size)
    cv2.imshow("3D Bounding Box", img_with_3dbbox)
    cv2.waitKey(0)
    cv2.destroyAllWindows()