# %% ########################
# 模块1：动态窗格生成器 modules/pane_generator.py
#############################
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors


class DynamicPaneGenerator:
    def __init__(self, config):
        self.num_panes = config['num_panes']
        self.emb_dim = config['embedding_size']
        self.online_kmeans = MiniBatchKMeans(n_clusters=self.num_panes)
        self.knn = NearestNeighbors(n_neighbors=5)

        # 初始化可学习的相似度矩阵
        self.sim_weight = torch.nn.Parameter(torch.eye(self.emb_dim))
        self.sim_bias = torch.nn.Parameter(torch.zeros(1))

    def _adaptive_similarity(self, embeddings):
        """可学习的相似度计算"""
        embeddings = torch.FloatTensor(embeddings)
        sim = torch.matmul(embeddings, self.sim_weight)
        sim = torch.matmul(sim, embeddings.T) + self.sim_bias
        return torch.sigmoid(sim).detach().cpu().numpy()

    def update_panes(self, user_embeddings, method='online_kmeans'):
        """更新窗格划分"""
        # 转换为numpy数组
        embeddings_np = user_embeddings.detach().cpu().numpy()

        if method == 'online_kmeans':
            self.online_kmeans.partial_fit(embeddings_np)
            pane_labels = self.online_kmeans.predict(embeddings_np)
        elif method == 'spectral':
            adj_matrix = self._adaptive_similarity(embeddings_np)
            sc = SpectralClustering(n_clusters=self.num_panes,
                                    affinity='precomputed')
            pane_labels = sc.fit_predict(adj_matrix)

        # 构建窗格成员关系
        pane_members = {}
        for idx, label in enumerate(pane_labels):
            if label not in pane_members:
                pane_members[label] = []
            pane_members[label].append(idx)

        return pane_labels, pane_members

    def get_pane_centers(self):
        """获取当前窗格中心"""
        return torch.FloatTensor(self.online_kmeans.cluster_centers_)

    def get_neighbor_mask(self, embeddings):
        """生成邻居掩码矩阵"""
        self.knn.fit(embeddings)
        distances, indices = self.knn.kneighbors(embeddings)
        mask = np.zeros((len(embeddings), len(embeddings)))
        for i, neighbors in enumerate(indices):
            mask[i, neighbors] = 1
        return torch.BoolTensor(mask)