
"""
    input:
        transform: (3, 4), numpy.ndarray, 包含(3, 3)的旋转矩阵rotation与(3, 1)的位移向量
    output:
        transform_inv (4, 4) numpy.ndarray, transform对应变换的逆变换矩阵
"""
def transform_inv(transform_input):
    rotation = transform_input[:3, :3]
    translation = transform_input[:3, 3]
    transform_inv = np.identity(4, dtype=np.double)
    rotation_inv = np.linalg.inv(rotation)
    transform_inv[:3, :3] = rotation_inv
    transform_inv[:3, 3] = rotation_inv @ ((-1.0) * translation)
    return transform_inv[:3, :]