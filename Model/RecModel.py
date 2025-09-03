from torch import nn

from Layer.BGA import BGA
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from scipy.sparse import coo_matrix, diags, csr_matrix
import numpy as np

from recbole.trainer import Trainer
from sklearn.cluster import KMeans

class GraphTransformer(nn.Module):
    def __init__(self, embedding_size):
        super(GraphTransformer, self).__init__()
        self.embedding_size = embedding_size
        # 这里可以定义多层Graph Transformer的层
        self.gloabal_layers = EncoderLayer(embedding_size, 4, 0.5)

    def forward(self, user_embeddings, item_embeddings):
        # 实现Graph Transformer的前向传播
        global_item_embeddings = self.gloabal_layers(user_embeddings, item_embeddings)

        return global_item_embeddings

class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count

        self.keys = nn.Linear(in_channels, key_channels)
        self.queries = nn.Linear(in_channels, key_channels)
        self.reprojection = nn.Linear(key_channels*head_count, key_channels)

    def forward(self, user_embeddings, item_embeddings):
        keys = self.keys(user_embeddings)
        queries = self.queries(item_embeddings)
        head_key_channels = self.key_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:,i * head_key_channels: (i + 1) * head_key_channels], dim=0)
            query = F.softmax(queries[:,i * head_key_channels: (i + 1) * head_key_channels], dim=1)
            context = F.relu((key @ query.transpose(0, 1))-0.1)
            attended_value = context.t() @ user_embeddings
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        return attention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.effectattn = EfficientAttention(in_channels = d_model, key_channels = 32, head_count =heads, value_channels = 32)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, y):
        x_pre = self.effectattn(x, y)
        y = y + self.dropout_1(x_pre)
        return y

def scipy_csr_to_torch_sparse_tensor(scipy_csr_mat):
    coo = scipy_csr_mat.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = coo.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))


class RecTransformer_norm(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset,head,layers,n_head,user_patch,item_patch,use_patch_attn=False):
        super(RecTransformer_norm, self).__init__(config, dataset)
        self.head = head
        self.user_patch = user_patch
        self.item_patch = item_patch
        self.epochs = 500
        # 模型参数初始化
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        # 嵌入层定义
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Graph Transformer
        self.layers = layers
        self.n_head = n_head
        # self.gat = GAT(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        self.user_bga = nn.ModuleList([BGA(self.n_items, self.embedding_size, self.embedding_size, self.embedding_size, layers, n_head,
                                         use_patch_attn)])
        self.item_bga = nn.ModuleList([BGA(self.n_items, self.embedding_size, self.embedding_size, self.embedding_size, layers, n_head,
                                         use_patch_attn)])

        for _ in range(1, self.n_layers):
            self.user_bga.append(BGA(self.n_items, self.embedding_size, self.embedding_size, self.embedding_size, layers, n_head,
                                         use_patch_attn))
            self.item_bga.append(BGA(self.n_items, self.embedding_size, self.embedding_size, self.embedding_size, layers, n_head,
                                         use_patch_attn))

        self.attn = None
        # Graph Transformer
        # self.in_mlp = nn.Linear(self.embedding_size,self.embedding_size)
        # self.graph_transformer = GraphTransformer(self.embedding_size)
        # 初始化权重
        self.apply(xavier_uniform_initialization)
        # 在这里初始化BPRLoss
        self.loss = BPRLoss()
        self.showEpoch = True
        # 指定输入类型
        self.device = config['device']
        # self.device = config['cpu']
        self.to(self.device)

        # 构建归一化的邻接矩阵
        self.norm_adj_matrix = self.get_norm_adj_mat(dataset)
        self.inter_matrix = scipy_csr_to_torch_sparse_tensor(dataset.inter_matrix(form='csr')).to(self.device)
        self.norm_adj_matrix = self.norm_adj_matrix.to(self.device)

    def forward(self,epoch=500):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)

        embeddings_list = [all_embeddings]
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            # user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
            #
            all_embeddings2 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            user_emb2, item_emb2 = torch.split(all_embeddings2, [self.n_users, self.n_items])
            if epoch>80:
                user_emb2 = self.user_bga[layer](user_emb2, self.user_patch)
                # item_emb2 = self.user_bga[layer](item_emb2, self.user_patch)
            all_embeddings = torch.cat([user_emb2, item_emb2], dim=0)

            embeddings_list.append(all_embeddings)


        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)  # + self.headView(epoch)

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, user_emb, item_emb


    def get_norm_adj_mat(self, dataset):

        n_nodes = self.n_users + self.n_items
        row = np.concatenate(
            [dataset.inter_matrix(form='coo').row, dataset.inter_matrix(form='coo').col + self.n_users])
        col = np.concatenate(
            [dataset.inter_matrix(form='coo').col + self.n_users, dataset.inter_matrix(form='coo').row])
        data = np.ones(len(row))

        adj_matrix = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(row_sum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_matrix)
        # Assuming norm_adj_tmp and d_mat_inv are defined earlier
        norm_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # Convert to COO format if not already in COO
        if not scipy.sparse.isspmatrix_coo(norm_adj_matrix):
            norm_adj_matrix = norm_adj_matrix.tocoo()

        # Construct sparse tensor in PyTorch
        row = torch.LongTensor(norm_adj_matrix.row)
        col = torch.LongTensor(norm_adj_matrix.col)
        data = torch.FloatTensor(norm_adj_matrix.data)
        shape = torch.Size(norm_adj_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(torch.stack([row, col]), data, shape)
        return sparse_tensor

    def get_norm_rawadj_mat(self, dataset):

        n_nodes = self.n_users + self.n_items
        row = dataset.inter_matrix(form='coo').row
        col = dataset.inter_matrix(form='coo').col
        data = np.ones(len(row))

        adj_matrix = coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(row_sum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_matrix)
        # Assuming norm_adj_tmp and d_mat_inv are defined earlier
        norm_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # Convert to COO format if not already in COO
        if not scipy.sparse.isspmatrix_coo(norm_adj_matrix):
            norm_adj_matrix = norm_adj_matrix.tocoo()

        # Construct sparse tensor in PyTorch
        row = torch.LongTensor(norm_adj_matrix.row)
        col = torch.LongTensor(norm_adj_matrix.col)
        data = torch.FloatTensor(norm_adj_matrix.data)
        shape = torch.Size(norm_adj_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(torch.stack([row, col]), data, shape)
        return sparse_tensor

    def calculate_loss(self, interaction,epoch):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb,user_in,item_in = self.forward(epoch)

        # 1. 计算预测分数矩阵 P (用户-商品的内积)
        P = torch.exp(torch.matmul(user_in, item_in.t())/1.5)  # 大小为 100x200

        # 2. 获取 A 中非零元素的索引（局部结构）
        A_indices = self.inter_matrix._indices()
        A_values = self.inter_matrix._values()

        # 3. 计算预测分数与邻接矩阵A的点积（只对A的非零元素计算）
        P_A = P[A_indices[0], A_indices[1]]  # 取出 P 中对应于 A 中非零元素的预测分数
        # 4. 计算损失
        # epsilon = 1e-8  # 避免数值不稳定
        # numerator = torch.sum(P_A * A_values)  # 分子：预测分数与 A 中对应的交互
        # denominator = torch.sum(P)+epsilon  # 分母：所有预测分数的和
        # local_loss = -torch.log(torch.clamp(numerator / denominator, min=epsilon)).mean()
        # print("localloss:"+str(local_loss))
        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item]
        neg_item_emb = item_emb[neg_item]

        # 计算正负样本得分
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        # 计算BPR损失
        loss = self.loss(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb,_,_= self.forward()

        user_emb = user_emb[user]
        item_emb = item_emb[self.head]

        # 计算预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def recall_at_k(self,pred_scores, ground_truth, k):
        """
        Calculate recall@k given predicted scores and ground truth.

        Parameters:
        - pred_scores: List or numpy array of predicted scores for items.
        - ground_truth: List or numpy array indicating ground truth labels (1 for relevant, 0 for non-relevant).
        - k: Top-k value.

        Returns:
        - recall: Recall@k value.
        """
        # 对预测分数进行排序，并获取top-k的索引
        top_k_indices = np.argsort(pred_scores)[::-1][:k]

        # 计算在top-k中实际为1（即真实为relevant的item）的数量
        num_relevant_in_top_k = np.sum(ground_truth[top_k_indices] == 1)

        # 计算总的relevant item数量
        total_relevant = np.sum(ground_truth == 1)

        # 计算recall@k
        recall = num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        return recall

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb,_,_ = self.forward()

        user_emb = user_emb[user]

        # 计算全排序预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

class RecTransformer(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset,head,layers,n_head,patch, patch2, use_patch_attn=False):
        super(RecTransformer, self).__init__(config, dataset)
        self.head = head
        self.patch = patch
        self.patch2 = patch2
        self.epochs = 500
        # 模型参数初始化
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        # 嵌入层定义
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Graph Transformer
        self.layers = layers
        self.n_head = n_head
        # self.gat = GAT(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        self.bga = nn.ModuleList([BGA(self.n_items, self.embedding_size, self.embedding_size, self.embedding_size, layers, n_head,
                                         use_patch_attn)])
        # self.graph_transformer = nn.ModuleList([GraphTransformer(self.embedding_size)])

        for _ in range(1, self.n_layers):
            self.bga.append(BGA(self.n_items, self.embedding_size, self.embedding_size, self.embedding_size, layers, n_head,
                                         use_patch_attn))
            # self.graph_transformer.append(GraphTransformer(self.embedding_size))

        self.attn = None
        # Graph Transformer
        # self.in_mlp = nn.Linear(self.embedding_size,self.embedding_size)
        # self.graph_transformer = GraphTransformer(self.embedding_size)
        # 初始化权重
        self.apply(xavier_uniform_initialization)
        # 在这里初始化BPRLoss
        self.loss = BPRLoss()
        self.showEpoch = True
        # 指定输入类型
        self.device = config['device']
        # self.device = config['cpu']
        self.to(self.device)

        # 构建归一化的邻接矩阵
        self.norm_adj_matrix = self.get_norm_adj_mat(dataset)
        self.inter_matrix = scipy_csr_to_torch_sparse_tensor(dataset.inter_matrix(form='csr')).to(self.device)
        self.norm_adj_matrix = self.norm_adj_matrix.to(self.device)

    def get_norm_adj_mat(self, dataset):

        n_nodes = self.n_users + self.n_items
        row = np.concatenate(
            [dataset.inter_matrix(form='coo').row, dataset.inter_matrix(form='coo').col + self.n_users])
        col = np.concatenate(
            [dataset.inter_matrix(form='coo').col + self.n_users, dataset.inter_matrix(form='coo').row])
        data = np.ones(len(row))

        adj_matrix = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(row_sum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_matrix)
        # Assuming norm_adj_tmp and d_mat_inv are defined earlier
        norm_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # Convert to COO format if not already in COO
        if not scipy.sparse.isspmatrix_coo(norm_adj_matrix):
            norm_adj_matrix = norm_adj_matrix.tocoo()

        # Construct sparse tensor in PyTorch
        row = torch.LongTensor(norm_adj_matrix.row)
        col = torch.LongTensor(norm_adj_matrix.col)
        data = torch.FloatTensor(norm_adj_matrix.data)
        shape = torch.Size(norm_adj_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(torch.stack([row, col]), data, shape)
        return sparse_tensor

    def get_norm_rawadj_mat(self, dataset):

        n_nodes = self.n_users + self.n_items
        row = dataset.inter_matrix(form='coo').row
        col = dataset.inter_matrix(form='coo').col
        data = np.ones(len(row))

        adj_matrix = coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv = np.power(row_sum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_matrix)
        # Assuming norm_adj_tmp and d_mat_inv are defined earlier
        norm_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # Convert to COO format if not already in COO
        if not scipy.sparse.isspmatrix_coo(norm_adj_matrix):
            norm_adj_matrix = norm_adj_matrix.tocoo()

        # Construct sparse tensor in PyTorch
        row = torch.LongTensor(norm_adj_matrix.row)
        col = torch.LongTensor(norm_adj_matrix.col)
        data = torch.FloatTensor(norm_adj_matrix.data)
        shape = torch.Size(norm_adj_matrix.shape)

        sparse_tensor = torch.sparse_coo_tensor(torch.stack([row, col]), data, shape)
        return sparse_tensor

    def headView(self,t):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        if t/self.epochs > 0:
            norm_adj = self.norm_adj_matrix
            # # 将 list 转换为 PyTorch 张量
            # head_indices = self.head
            # # tail_indices = torch.tensor(self.tail)
            #
            # norm_item = F.normalize(item_emb)
            # cosine_similarity = norm_item[head_indices] @ norm_item[head_indices].t()
            #
            # # 找出余弦相似度大于0.8的索引
            # # adj_matrix = torch.zeros((self.n_items, self.n_items)).to(self.device)
            # mask = cosine_similarity > self.a
            # cosine_similarity[cosine_similarity>0]=1
            # # 只保留符合条件的位置
            # masked_similarity = cosine_similarity * mask
            #
            # # 获取非零元素的行和列索引
            # head_nonzero_idx, tail_nonzero_idx = torch.nonzero(masked_similarity, as_tuple=True)
            #
            # # 获取这些非零元素的值
            # values = masked_similarity[head_nonzero_idx, tail_nonzero_idx]
            #
            # # 调整这些索引的维度以对称化邻接矩阵
            # indices = torch.cat([
            #     torch.stack([head_indices[head_nonzero_idx], head_indices[tail_nonzero_idx]]),  # 原始的非零项
            #     torch.stack([head_indices[tail_nonzero_idx], head_indices[head_nonzero_idx]])  # 对称项
            # ], dim=1)
            #
            # # 拼接对应的值来形成完整的邻接矩阵
            # values = torch.cat([values, values])
            #
            # # 构建稀疏邻接矩阵
            # adj_matrix = torch.sparse_coo_tensor(indices, values, (self.n_items, self.n_items)).to(self.device)
            #
            # norm_adj = self.update_and_normalize_adj_matrix(self.norm_adj,adj_matrix)
        else:
            norm_adj = self.norm_adj_matrix
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings]

        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)

        return final_embeddings

    def trans_inter_matrix(self, inter_matrix):

        # 获取 A 的索引和尺寸
        original_indices = inter_matrix._indices()
        original_values = self.norm_adj_matrix._values()
        original_size = inter_matrix.size()
        # 转置索引：交换索引的两个维度
        transposed_indices = original_indices[[1, 0], :]
        # 调整尺寸：交换尺寸的两个维度
        transposed_size = torch.Size([original_size[1], original_size[0]])
        # 创建转置后的稀疏张量
        inter_matrix_T = torch.sparse.FloatTensor(transposed_indices, original_values, transposed_size)
        return inter_matrix_T

    def CFChannel(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)

        embeddings_list = [all_embeddings]
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            # user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
            #
            all_embeddings2 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            user_emb2, item_emb2 = torch.split(all_embeddings2, [self.n_users, self.n_items])
            all_embeddings = torch.cat([user_emb2, item_emb2], dim=0)

            embeddings_list.append(all_embeddings)


        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)  # + self.headView(epoch)

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def firstStage(self,user_emb, item_emb, epoch):
        # # item_emb2 = self.graph_transformer(user_emb,item_emb)
        # all_embeddings = torch.cat([user_emb2, item_emb2], dim=0)

        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings]
        user_emb2 = user_emb.clone()
        item_emb2 = item_emb.clone()
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            # user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
            #
            # all_embeddings = torch.cat([user_emb, item_emb], dim=0)

            # all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            # user_emb2, item_emb2 = torch.split(all_embeddings, [self.n_users, self.n_items])
            # if epoch>200:
            #     item_emb2 = self.bga[layer](item_emb2, self.patch)

            # user_embO = self.graph_transformer[layer](item_emb2, user_emb2)
            all_embeddings2 = self.bga[layer](all_embeddings, self.patch)
            # item_emb2 = torch.sparse.mm(inter_matrix_T, user_emb2)
            all_embeddings3 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            _, item_embO = torch.split(all_embeddings3, [self.n_users, self.n_items])
            user_embO, _ = torch.split(all_embeddings2, [self.n_users, self.n_items])
            all_embeddings = torch.cat([user_embO, item_embO], dim=0)

            # user_emb2, item_emb2 = torch.split(all_embeddings, [self.n_users, self.n_items])

            embeddings_list.append(all_embeddings)

        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)  # + self.headView(epoch)

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        if epoch > 100000 and epoch % 50 == 0:
            print(epoch)
            score_matrix = F.normalize(user_emb) @ F.normalize(item_emb.T)

            # 定义阈值 score_matrix[:,self.head].max()
            scores = score_matrix[:, self.head]

            # # 确定前 20% 的数量
            # num_items = scores.size(1)
            # top_20_percent = 1
            #
            # # 对每行（用户的物品评分）进行排序，并选择前 20% 的评分
            # top_scores, _ = torch.topk(scores, top_20_percent, dim=1)

            # 计算每个用户前 20% 分数的均值
            threshold = torch.max(scores, 1, keepdim=True)[0]

            # 找到大于阈值的位置
            score_matrix[:, self.head] = 0
            mask = score_matrix > threshold.expand(-1, score_matrix.shape[1])  # 布尔张量，大小为 [n_users, n_items]
            print("add:" + str(mask.sum()))
            score_matrix[score_matrix > 0] = 1
            # 获取满足条件的行索引和列索引
            new_edge_indices = mask.nonzero(as_tuple=False).t()  # 大小为 [2, num_new_edges]
            if mask.sum() > 0:
                # 获取对应的分数值
                new_edge_values = score_matrix[mask]  # 大小为 [num_new_edges]

                # 创建新的边的稀疏张量
                new_edges_sparse = torch.sparse.FloatTensor(new_edge_indices, new_edge_values,
                                                            torch.Size([self.n_users, self.n_items])).to(self.device)

                # 将新的边添加到原始交互矩阵中
                updated_interaction_sparse = self.inter_matrix + new_edges_sparse

                # 如果存在重复边，可以将值限制在 1.0
                # 在访问 values 之前先调用 coalesce()
                updated_interaction_sparse = updated_interaction_sparse.coalesce()

                # 然后再访问 values
                updated_values = torch.clamp(updated_interaction_sparse.values(), max=1.0)
                updated_interaction_sparse = torch.sparse.FloatTensor(updated_interaction_sparse.indices(),
                                                                      updated_values,
                                                                      updated_interaction_sparse.size())
                # self.inter_matrix = updated_interaction_sparse

                # 确保稀疏张量是合并的
                if not updated_interaction_sparse.is_coalesced():
                    updated_interaction_sparse = updated_interaction_sparse.coalesce()
                # 获取非零元素的索引
                indices = updated_interaction_sparse.indices()  # 形状为 [2, nnz]
                n_nodes = self.n_users + self.n_items
                # 行索引和列索引
                row_indices = indices[0]  # 行索引
                col_indices = indices[1]  # 列索引
                user_indices = row_indices
                item_indices = col_indices + self.n_users
                row = torch.cat([user_indices, item_indices])
                col = torch.cat([item_indices, user_indices])

                # Create data tensor (all ones)
                data = torch.ones(len(row), dtype=torch.float32, device=self.device)

                # # Create sampled adjacency matrix in COO format using PyTorch
                sampled_adj_matrix = torch.sparse_coo_tensor(
                    torch.stack([row, col]),
                    data,
                    (n_nodes, n_nodes),
                    device=self.device
                )

                # Sum rows (degree of nodes)
                row_sum = torch.sparse.sum(sampled_adj_matrix, dim=1).to_dense()

                # Inverse square root of the row sum
                d_inv = torch.pow(row_sum, -0.5)
                d_inv[torch.isinf(d_inv)] = 0.0

                # Create sparse diagonal matrix for D^(-1/2)
                d_inv_indices = torch.arange(n_nodes, device=self.device)
                d_inv_diag = torch.sparse_coo_tensor(
                    torch.stack([d_inv_indices, d_inv_indices]),
                    d_inv,
                    (n_nodes, n_nodes),
                    device=self.device
                )

                # Normalize the adjacency matrix using sparse matrix multiplication
                norm_adj_tmp = torch.sparse.mm(d_inv_diag, sampled_adj_matrix)  # D^(-1/2) * A
                norm_adj_matrix = torch.sparse.mm(norm_adj_tmp, d_inv_diag)  # (D^(-1/2) * A) * D^(-1/2)
                self.norm_adj_matrix = norm_adj_matrix
        return user_all_embeddings, item_all_embeddings

    def secondStage(self,user_emb, item_emb):
        # # item_emb2 = self.graph_transformer(user_emb,item_emb)
        # all_embeddings = torch.cat([user_emb2, item_emb2], dim=0)

        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings]
        user_emb2 = user_emb.clone()
        item_emb2 = item_emb.clone()
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            # user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
            #
            # all_embeddings = torch.cat([user_emb, item_emb], dim=0)

            # all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            # user_emb2, item_emb2 = torch.split(all_embeddings, [self.n_users, self.n_items])
            # if epoch>200:
            #     item_emb2 = self.bga[layer](item_emb2, self.patch)

            # user_embO = self.graph_transformer[layer](item_emb2, user_emb2)
            all_embeddings2 = self.bga[layer](all_embeddings, self.patch)
            # item_emb2 = torch.sparse.mm(inter_matrix_T, user_emb2)
            all_embeddings3 = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            user_embO, _ = torch.split(all_embeddings3, [self.n_users, self.n_items])
            _, item_embO = torch.split(all_embeddings2, [self.n_users, self.n_items])
            all_embeddings = torch.cat([user_embO, item_embO], dim=0)

            # # item_embO = self.graph_transformer[layer](user_emb2, item_emb2)
            # item_embO = self.bga[layer](item_emb2, user_emb2, self.patch2, self.patch)
            # # item_emb2 = torch.sparse.mm(inter_matrix_T, user_emb2)
            # all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            # user_embO, _ = torch.split(all_embeddings, [self.n_users, self.n_items])
            # all_embeddings = torch.cat([user_embO, item_embO], dim=0)
            #
            # user_emb2, item_emb2 = torch.split(all_embeddings, [self.n_users, self.n_items])

            embeddings_list.append(all_embeddings)

        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)  # + self.headView(epoch)

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings

    def forward(self,epoch=500):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        user_all_embeddings2, item_all_embeddings2 = self.CFChannel()
        if epoch>1000:
            user_all_embeddings, item_all_embeddings = self.secondStage(user_emb, item_emb)
        else:
            user_all_embeddings, item_all_embeddings = self.firstStage(user_emb, item_emb, epoch)

        user_all_embeddings = 0.8*user_all_embeddings+0.2*user_all_embeddings2
        item_all_embeddings = 0.8*item_all_embeddings+0.2*item_all_embeddings2

        return user_all_embeddings, item_all_embeddings, user_emb, item_emb

    def calculate_loss(self, interaction,epoch):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_emb, item_emb,user_in,item_in = self.forward(epoch)

        # 1. 计算预测分数矩阵 P (用户-商品的内积)
        P = torch.exp(torch.matmul(user_in, item_in.t())/1.5)  # 大小为 100x200

        # 2. 获取 A 中非零元素的索引（局部结构）
        A_indices = self.inter_matrix._indices()
        A_values = self.inter_matrix._values()

        # 3. 计算预测分数与邻接矩阵A的点积（只对A的非零元素计算）
        P_A = P[A_indices[0], A_indices[1]]  # 取出 P 中对应于 A 中非零元素的预测分数
        # 4. 计算损失
        # epsilon = 1e-8  # 避免数值不稳定
        # numerator = torch.sum(P_A * A_values)  # 分子：预测分数与 A 中对应的交互
        # denominator = torch.sum(P)+epsilon  # 分母：所有预测分数的和
        # local_loss = -torch.log(torch.clamp(numerator / denominator, min=epsilon)).mean()
        # print("localloss:"+str(local_loss))
        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item]
        neg_item_emb = item_emb[neg_item]

        # 计算正负样本得分
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        # 计算BPR损失
        loss = self.loss(pos_score, neg_score)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb,_,_= self.forward()

        user_emb = user_emb[user]
        item_emb = item_emb[self.head]

        # 计算预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def recall_at_k(self,pred_scores, ground_truth, k):
        """
        Calculate recall@k given predicted scores and ground truth.

        Parameters:
        - pred_scores: List or numpy array of predicted scores for items.
        - ground_truth: List or numpy array indicating ground truth labels (1 for relevant, 0 for non-relevant).
        - k: Top-k value.

        Returns:
        - recall: Recall@k value.
        """
        # 对预测分数进行排序，并获取top-k的索引
        top_k_indices = np.argsort(pred_scores)[::-1][:k]

        # 计算在top-k中实际为1（即真实为relevant的item）的数量
        num_relevant_in_top_k = np.sum(ground_truth[top_k_indices] == 1)

        # 计算总的relevant item数量
        total_relevant = np.sum(ground_truth == 1)

        # 计算recall@k
        recall = num_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        return recall

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        user_emb, item_emb,_,_ = self.forward()

        user_emb = user_emb[user]

        # 计算全排序预测评分
        scores = torch.matmul(user_emb, item_emb.t())
        return scores