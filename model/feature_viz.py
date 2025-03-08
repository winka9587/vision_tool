import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def vis_feature_map(featrue_map):
    '''
    将二维特征图可视化为RGB图像

    Parameters:
        featrue_map (numpy.array): 维度为[h, w, feat_dim]的特征图

    Returns:
        None: 直接可视化出来
    '''
    # 特征降维 对应RGB通道
    pca = PCA(n_components=3)   
    reshaped_feature_map = np.reshape(featrue_map, (-1, featrue_map.shape[-1])) # [patch, patch, feat_dim] -> [patch*patch, feat_dim]
    reduced_feature_map = pca.fit_transform(reshaped_feature_map)   # [patch*patch, feat_dim] -> [patch*patch, 3]
    reconstructed_feature_map = np.reshape(reduced_feature_map, (featrue_map.shape[0], featrue_map.shape[1], 3))    # [patch*patch, 3] -> [patch, patch, 3]

    # 归一化
    min_val = np.min(reconstructed_feature_map)
    max_val = np.max(reconstructed_feature_map)
    normalized_features = 255 * (reconstructed_feature_map - min_val) / (max_val - min_val)

    # 可视化
    rgb = np.uint8(normalized_features)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()