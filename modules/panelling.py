# %% ########################
# 模块1：动态窗格管理 modules/panelling.py
#############################
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans, Birch, SpectralClustering
from collections import defaultdict

from torch import cosine_similarity

# from modules.llm_enhancer import PaneLLMEnhancer


class DynamicPaneManager:
    def __init__(self, config, dataset):
        self.num_panes = config['num_panes']
        self.emb_dim = config['embedding_size']
        self.kmeans = Birch(n_clusters=self.num_panes)
        # self.llm_enhancer = PaneLLMEnhancer(dataset, config)
        self.pane_centers = None
        self.pane_labels = None

    def _affinity_matrix(self, embeddings):
        """构建自适应相似度矩阵"""
        import torch.nn.functional as F
        # 计算可学习相似度
        embeddings = F.normalize(embeddings)
        sim = torch.matmul(embeddings, embeddings.T)
        return torch.sigmoid(sim).detach().cpu().numpy()

    def update_panes(self, embeddings,inter_data):
        # LLms更新
        """1. 评估当前分组，获得前3个最大交互的商品索引."""
        """2. 设定训练模型更新周期，获得当前周期每个用户的推荐列表"""
        """3. 评估与当前分组交互画像不太相符的商品（最大交互的商品与组的交互情况不太相符），将这些商品的交互情况与每组的交互情况输入给LLMs让它推理这些商品该属于那一组"""
        """4. 获得新的分组列表"""
        # 计算相似度矩阵
        adj_matrix = self._affinity_matrix(embeddings)
        """更新窗格划分"""
        sc = SpectralClustering(
            n_clusters=self.num_panes,
            affinity='precomputed',
            assign_labels='discretize'
        )
        self.pane_labels = sc.fit_predict(adj_matrix)
        embeddings_np = embeddings.detach().cpu().numpy()
        # # 增量聚类
        # self.kmeans.partial_fit(embeddings_np)
        #
        # # 获取标签并计算中心
        # self.pane_labels = self.kmeans.predict(embeddings_np)
        unique_labels = np.unique(self.pane_labels)

        centers = []
        for label in unique_labels:
            cluster_emb = embeddings_np[self.pane_labels == label]
            centers.append(cluster_emb.mean(axis=0))
        self.pane_centers = torch.tensor(
            np.array(centers),
            device=embeddings.device
        )

        # 构建窗格成员关系
        pane_members = {}
        for idx, label in enumerate(self.pane_labels):
            if label not in pane_members:
                pane_members[label] = []
            pane_members[label].append(idx)
            # 构建用户到商品的映射
            user_item_dict = {}
        # 假设inter_data是包含'user_id'和'item_id'列的DataFrame
        for _, row in inter_data.iterrows():
            user = int(row['user_id'])
            item = int(row['item_id'])
            if user not in user_item_dict:
                user_item_dict[user] = set()
            user_item_dict[user].add(item)

        # 生成每个窗格的商品组成员（pane_item_members）
        pane_item_members = {}
        for pane_label, users in pane_members.items():
            items = set()
            for user in users:
                items.update(user_item_dict.get(user, set()))
            pane_item_members[pane_label] = list(items)

        return self.pane_labels, pane_members, pane_item_members

    def generate_group_full_connect_tensor(self,pane_members, pane_item_members):
        """使用张量广播优化全连接标记"""
        # 获取所有用户和商品ID
        all_users = list({u for pane in pane_members.values() for u in pane})
        all_items = list({i for pane in pane_item_members.values() for i in pane})

        # 创建映射
        user_to_idx = {u: i for i, u in enumerate(all_users)}
        item_to_idx = {i: j for j, i in enumerate(all_items)}

        num_panes = len(pane_members)
        num_users = len(all_users)
        num_items = len(all_items)

        # 转换组成员为索引列表
        user_index_list = [
            [user_to_idx[u] for u in pane_members[pane_id]]
            for pane_id in sorted(pane_members.keys())
        ]
        item_index_list = [
            [item_to_idx[i] for i in pane_item_members[pane_id]]
            for pane_id in sorted(pane_item_members.keys())
        ]

        # 确定最大成员数（用户和商品独立处理）
        max_users = max(len(users) for users in user_index_list)
        max_items = max(len(items) for items in item_index_list)
        pad_value = -1
        # 填充或截断索引列表
        def pad_or_truncate(indices, max_len):
            return (
                indices[:max_len] + [pad_value] * (max_len - len(indices))
                if len(indices) < max_len
                else indices[:max_len]
            )

        padded_user_indices = [
            pad_or_truncate(users, max_users) for users in user_index_list
        ]
        padded_item_indices = [
            pad_or_truncate(items, max_items) for items in item_index_list
        ]

        # 转换为张量
        user_tensor = torch.tensor(padded_user_indices, dtype=torch.long)
        item_tensor = torch.tensor(padded_item_indices, dtype=torch.long)

        # 初始化三维张量
        tensor = torch.zeros(num_panes, num_users, num_items)

        for pane_id in pane_members:
            # 获取当前组的用户/商品索引
            user_indices = torch.tensor(
                [user_to_idx[u] for u in pane_members[pane_id]],
                dtype=torch.long
            )
            item_indices = torch.tensor(
                [item_to_idx[i] for i in pane_item_members[pane_id]],
                dtype=torch.long
            )

            if len(user_indices) == 0 or len(item_indices) == 0:
                continue

            # 关键修正：直接使用广播索引
            tensor[pane_id, user_indices[:, None], item_indices[None, :]] = 1.0

        return tensor, user_tensor, item_tensor

    def get_pane_mask(self, embeddings):
        """生成窗格注意力掩码"""
        pane_mask = torch.zeros(
            embeddings.size(0),
            embeddings.size(0),
            device=embeddings.device
        )
        for k in range(self.num_panes):
            indices = torch.where(torch.tensor(
                self.pane_labels == k,
                device=embeddings.device
            ))[0]
            pane_mask[indices[:, None], indices] = 1
        return pane_mask.bool()

    def get_pane_embeddings(self):
        """获取当前窗格中心嵌入"""
        return self.pane_centers

    def update_groups(self, user_embeddings, inter_data):
        """动态更新用户分组"""
        # 阶段1：基础分组更新
        self._basic_clustering(user_embeddings)

        # 阶段2：分组特征分析
        group_features = self._analyze_group_features(inter_data)

        # 阶段3：异常用户检测
        anomalous_users = self._detect_anomalous_users(inter_data, group_features)
        self._llm_based_reassignment(anomalous_users, group_features,inter_data)

        # 阶段4：结构更新
        self._update_group_centers(user_embeddings)
        # self.update_counter += 1

        return self.group_labels, self._build_group_members()

    def _basic_clustering(self, embeddings):
        """核心聚类逻辑"""
        # 增强相似度矩阵：70%嵌入相似度 + 30%行为相似度
        adj_matrix = self._affinity_matrix(embeddings)

        sc = SpectralClustering(
            n_clusters=self.num_panes,
            affinity='precomputed',
            assign_labels='discretize'
        )
        self.group_labels = sc.fit_predict(adj_matrix)

    def _embedding_similarity(self, embeddings):
        """计算用户嵌入余弦相似度"""
        return cosine_similarity(embeddings.detach().cpu().numpy())

    # def _behavior_similarity(self):
    #     """计算基于交互行为的Jaccard相似度"""
    #     # 需要预加载用户行为数据
    #     # 返回n_users x n_users的相似度矩阵
    #     pass

    def _analyze_group_features(self, inter_data):
        """分析各分组特征

        参数:
            inter_data: DataFrame, 包含交互数据，需有'user_id'和'item_id'列

        返回:
            包含各分组特征的字典，每个组包含:
            - top_items: 热门交互商品列表
            - rating_stats: 评分统计信息（如果有rating字段）
            - member_count: 组成员数量
        """
        group_features = {}

        for group in range(self.num_panes):
            # 获取组内用户
            members = np.where(self.group_labels == group)[0]

            # 筛选组内成员的交互记录
            member_interactions = inter_data[inter_data['user_id'].isin(members)]

            # 特征1：热门交互商品
            item_freq = member_interactions['item_id'].value_counts().reset_index()
            item_freq.columns = ['item_id', 'frequency']
            top_items = item_freq.sort_values('frequency', ascending=False)[:20]

            # 特征2：交互评分分析（如果有rating字段）
            rating_stats = {}
            if 'rating' in member_interactions.columns:
                # 组内整体评分统计
                rating_stats['group_avg_rating'] = member_interactions['rating'].mean()
                rating_stats['rating_distribution'] = member_interactions['rating'].value_counts().to_dict()

                # 热门商品的评分统计
                top_items_ratings = []
                for item_id in top_items['item_id']:
                    item_ratings = member_interactions[member_interactions['item_id'] == item_id]['rating']
                    if not item_ratings.empty:
                        top_items_ratings.append({
                            'item_id': item_id,
                            'avg_rating': item_ratings.mean()
                        })
                rating_stats['top_items_ratings'] = top_items_ratings

            group_features[group] = {
                'top_items': top_items['item_id'].tolist(),
                'rating_stats': rating_stats if rating_stats else None,
                'member_count': len(members),
                # 可以保留原有的活跃时段特征
                # 'active_pattern': avg_time,
            }
        return group_features

    def _detect_anomalous_users(self, inter_data, group_features):
        """检测行为异常用户"""
        anomalous_users = []

        # 按用户分组交互数据
        user_groups = inter_data.groupby('user_id')

        for user_id, user_interactions in user_groups:
            group = self.group_labels[user_id]

            # 获取用户交互过的商品集合
            user_items = set(user_interactions['item_id'])

            # 指标1：兴趣匹配度
            group_top_items = set(group_features[group]['top_items'])
            item_overlap = len(user_items & group_top_items)
            interest_score = item_overlap / len(group_top_items) if len(group_top_items) > 0 else 0

            # 判断是否为异常用户
            if interest_score < 0.2:
                anomalous_users.append(user_id)

                # 如果达到最大数量限制，提前终止
                if len(anomalous_users) >= 100:
                    break

        return anomalous_users # 限制处理数量 anomalous_users[:100]

    def _llm_based_reassignment(self, users, group_features,inter_data):
        """基于LLM的用户重分配"""
        for user_id in users:
            # current_group = self.group_labels[user_id]
            user_profile = self._build_user_profile(inter_data,user_id)

            prompt = f"Given the profile of a user: {user_profile}\nand the profiles of K user groups{group_features}\n, analyze and determine which group the user most likely belongs to. Each user belongs to one and only one group. Output only the group number (an integer between 1 and K)."
            response = self.llm_enhancer.query(prompt)

            try:
                new_group = int(response.strip())
                if 0 <= new_group < self.num_panes:
                    self.group_labels[user_id] = new_group
            except:
                pass

    def _build_user_profile(self, inter_data, user_id, recent_items_limit=20, top_items_limit=5):
        """基于DataFrame格式的交互数据构建用户特征描述

        参数:
            user_id: 目标用户ID
            inter_data: DataFrame, 包含交互数据，需有'user_id'和'item_id'列
            recent_items_limit: int, 保留的最近交互商品数量
            top_items_limit: int, 保留的高频交互商品数量

        返回:
            包含用户特征的字典
        """
        # 筛选该用户的交互记录
        user_interactions = inter_data[inter_data['user_id'] == user_id]

        # 如果没有交互记录，返回空特征
        if user_interactions.empty:
            return {
                'interacted_items': [],
                'total_interactions': 0,
                'unique_items': 0,
                'top_frequent_items': []
            }

        # 特征1：交互商品列表（按时间排序取最近，如果有timestamp字段）
        if 'timestamp' in user_interactions.columns:
            recent_items = user_interactions.sort_values('timestamp', ascending=False)['item_id'].tolist()
        else:
            recent_items = user_interactions['item_id'].tolist()
        item_list = recent_items[:recent_items_limit]

        # # 特征2：高频交互商品分析
        # item_freq = user_interactions['item_id'].value_counts().to_dict()
        # top_items = sorted(item_freq.items(), key=lambda x: -x[1])[:top_items_limit]
        keys=[3,4]
        # 特征3：交互评分分析（如果有rating字段）
        rating_stats = {}
        if 'rating' in user_interactions.columns:
            rating_dist = user_interactions['rating'].value_counts().to_dict()
            rating_stats = {
                'avg_rating': user_interactions['rating'].mean()
                # 'rating_distribution': {float(k): int(v) for k, v in sorted(rating_dist.items())}
            }

        # # 特征4：时间模式分析（如果有timestamp字段）
        # time_pattern = {}
        # if 'timestamp' in user_interactions.columns:
        #     time_pattern = {
        #         'first_interaction': user_interactions['timestamp'].min(),
        #         'last_interaction': user_interactions['timestamp'].max(),
        #         'interaction_count': len(user_interactions)
        #     }

        return {
            # 原始交互特征
            'interacted_items': item_list,

            # 统计特征
            # 'total_interactions': len(user_interactions),
            # 'unique_items': len(user_interactions['item_id'].unique()),
            # 'top_frequent_items': [{'item_id': k, 'count': v} for k, v in top_items],

            # 评分特征（可选）
            'rating_stats': rating_stats

            # # 时间特征（可选）
            # 'time_pattern': time_pattern,
            #
            # # 元信息
            # 'profile_generated_at': pd.Timestamp.now().isoformat()
        }

    def _update_group_centers(self, embeddings):
        """更新分组中心点"""
        centers = []

        for group in range(self.num_panes):
            mask = self.group_labels == group
            if sum(mask) == 0:
                centers.append(torch.mean(embeddings, dim=0))
            else:
                centers.append(torch.mean(embeddings[mask], dim=0))
        self.pane_centers = torch.tensor(
            np.array(centers),
            device=embeddings.device
        )

    def _build_group_members(self):
        """构建分组成员列表"""
        groups = defaultdict(list)
        for user_id, group in enumerate(self.group_labels):
            groups[group].append(user_id)
        return dict(groups)