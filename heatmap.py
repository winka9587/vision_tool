import copy
import open3d as o3d
import numpy as np

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_color_map(x):
    colours = plt.cm.Spectral(x)
    return colours[:, :3]

def embed_tsne(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_results = np.squeeze(tsne_results)
    tsne_min = np.min(tsne_results)
    tsne_max = np.max(tsne_results)
    return (tsne_results - tsne_min) / (tsne_max - tsne_min)


def embed_tsne_pair(data):
    """
    N x D np.array data
    """
    tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data)
    tsne_results_raw = np.squeeze(tsne_results)
    return tsne_results_raw


def embed_tsne_pair_new(data1, data2):
    """
    # 生成两组feat在同一尺度下的TSNE color
    Input:
        data1: N x D np.array data
        data2: M x D np.array data
    Output:
        tsne_results_raw: (M+N,) np.array data
        border_idx: 划分两组feat的边界 int
    """
    border_idx = data1.shape[0]
    data_emb = np.vstack((data1, data2))
    tsne = TSNE(n_components=1, verbose=1, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(data_emb)
    tsne_results_raw = np.squeeze(tsne_results)
    return tsne_results_raw, border_idx


def get_color_map_feature(feature):
    tsne_results = embed_tsne(feature)
    color = get_color_map(tsne_results)
    return color


# use same normalize parameter
# feature1, feature2: (N,C)
def get_color_map_feature_pair(feature1, feature2):
    tsne_results_raw, border_idx = embed_tsne_pair_new(feature1, feature2)
    tsne_min = np.min(tsne_results_raw)
    tsne_max = np.max(tsne_results_raw)
    tsne_results = (tsne_results_raw - tsne_min) / (tsne_max - tsne_min)
    color1 = get_color_map(tsne_results[:border_idx])
    color2 = get_color_map(tsne_results[border_idx:])
    return color1, color2


"""
    visualize two feature in same norm config to cmp
    pcd_1, pcd_2: (N, 3) numpy.ndarray
    feature1, feature2: (N,C) numpy.ndarray
    comment: str
"""


def viz_color_map_feat(pcd_1, pcd_2, feat_1, feat_2, comment=""):
    # use TSNE to visualize color map of feature
    color1_global, color2_global = get_color_map_feature_pair(feat_1.squeeze(0).cpu(),
                                                            feat_2.squeeze(0).cpu())
    import PointCloudRender
    pcr = PointCloudRender()
    pcr.render_multi_pts("test", [pcd_1, pcd_2 + np.array([0.15, 0.0, 0.0])], [color1_global, color2_global])
