import pandas as pd
import scipy
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import init_logger
from scipy.sparse import coo_matrix, diags
from torch import nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from scipy.sparse import coo_matrix, diags, csr_matrix
import numpy as np
import torch.nn.functional as F

from Layer.BGA import BGA
# from modules import *
# from modules.exploration import TransformerExplorer
# from modules.llm_enhancer import PaneLLMEnhancer
from modules.panelling import DynamicPaneManager


def get_norm_adj_mat( dataset,n_users,n_items):
    n_nodes = n_users + n_items
    row = np.concatenate(
        [dataset.inter_matrix(form='coo').row, dataset.inter_matrix(form='coo').col + n_users])
    col = np.concatenate(
        [dataset.inter_matrix(form='coo').col + n_users, dataset.inter_matrix(form='coo').row])
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


def scipy_csr_to_torch_sparse_tensor(scipy_csr_mat):
    coo = scipy_csr_mat.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = coo.shape
    return torch.sparse.FloatTensor(indices, values, torch.Size(shape))

class DLGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.data = dataset
        # 关键修正：直接使用Interaction的原始数据构建DataFrame
        self.inter_data = pd.DataFrame({
            'user_id': self.data.inter_feat[dataset.uid_field].numpy(),
            'item_id': self.data.inter_feat[dataset.iid_field].numpy(),
            # 如果有其他字段（如rating/timestamp）按需添加
            'rating': self.data.inter_feat['rating'].numpy() if 'rating' in self.data.inter_feat else None,
            'timestamp': self.data.inter_feat['timestamp'].numpy() if 'timestamp' in self.data.inter_feat else None
        })
        self.config = config
        # 基础参数
        self.embed_dim = config['embedding_size']
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.showEpoch = True
        self.user_patch = None
        self.item_patch = None
        # 初始化嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_dim)
        self.user_embedding2 = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding2 = nn.Embedding(self.n_items, self.embed_dim)
        self.last_pane_update_epoch=-1
        self.LR = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.user_embedding2.weight, std=0.1)
        nn.init.normal_(self.item_embedding2.weight, std=0.1)

        use_patch_attn = False
        # LightGCN卷积层
        self.n_layers = 3
        self.n_heads = 4
        self.user_bga = nn.ModuleList([BGA(self.n_items, self.embed_dim, self.embed_dim, self.embed_dim, self.n_layers, self.n_heads,
                                         use_patch_attn)])
        self.item_bga = nn.ModuleList([BGA(self.n_items, self.embed_dim, self.embed_dim, self.embed_dim, self.n_layers, self.n_heads,
                                         use_patch_attn)])

        for _ in range(1, self.n_layers):
            self.user_bga.append(BGA(self.n_items, self.embed_dim, self.embed_dim, self.embed_dim, self.n_layers, self.n_heads,
                                         use_patch_attn))
            self.item_bga.append(BGA(self.n_items, self.embed_dim, self.embed_dim, self.embed_dim, self.n_layers, self.n_heads,
                                         use_patch_attn))

        # 构建归一化的邻接矩阵
        self.norm_adj_matrix = get_norm_adj_mat(dataset, dataset.user_num, dataset.item_num)
        self.inter_matrix = scipy_csr_to_torch_sparse_tensor(dataset.inter_matrix(form='csr')).to(self.device)
        self.norm_adj_matrix = self.norm_adj_matrix.to(self.device)



        # 动态窗格组件
        self.pane_manager = DynamicPaneManager(config,dataset)
        # self.explorer = TransformerExplorer(self.embed_dim)
        # self.llm_enhancer = PaneLLMEnhancer(dataset, config)

        # 融合参数
        self.gate = nn.Linear(2 * self.embed_dim, 1)
        self.pane_embs = None
        self.to(torch.device('cuda'))

    def get_ego_embeddings(self):
        return torch.cat([self.user_embedding.weight,
                          self.item_embedding.weight])

    def attention_alignment_loss(self,att, adj):
        """
        计算多头注意力与邻接矩阵的对齐损失

        参数:
            att (Tensor): 多头注意力张量，形状为 (k, n, m)，其中k为注意力头数
            adj (Tensor): 二分图邻接矩阵，形状为 (n, m)，元素为0或1

        返回:
            loss (Tensor): 对齐损失值
        """

        # 将邻接矩阵转换为浮点型并处理可能的NaN值
        adj_float = adj.float().clamp(0.0, 1.0)  # 确保值在[0,1]之间
        adj_float2 = adj_float.clone()
        adj_float2[adj_float > 0] = 1
        # 计算二元交叉熵损失（带logits处理）
        loss = (att*adj_float2).sum()/att.sum()

        return loss

    def firstStage(self,user_emb, item_emb, epoch):
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = self.LR(all_embeddings)
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        cos = F.normalize(user_emb)@F.normalize(item_emb).T
        alignment_loss = self.attention_alignment_loss(cos,self.inter_matrix.to_dense())

        embeddings_list = [all_embeddings]
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            user_embO = self.user_bga[layer](all_embeddings, self.n_users, self.n_items, self.user_patch, self.item_patch)
            all_embeddings = torch.cat([user_embO, item_emb], dim=0)
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            # _, item_embO = torch.split(all_embeddings3, [self.n_users, self.n_items])
            # all_embeddings = torch.cat([user_embO, item_embO], dim=0)

            embeddings_list.append(all_embeddings)

        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)  # + self.headView(epoch)

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, alignment_loss

    def secondStage(self,user_emb, item_emb, epoch):
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = self.LR(all_embeddings)
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        cos = F.normalize(user_emb)@F.normalize(item_emb).T
        alignment_loss = self.attention_alignment_loss(cos,self.inter_matrix.to_dense())

        embeddings_list = [all_embeddings]
        # 消息传递与聚合过程
        for layer in range(self.n_layers):
            item_embO = self.item_bga[layer](all_embeddings, self.n_users, self.n_items, self.user_patch, self.item_patch, user_use=False)
            all_embeddings = torch.cat([user_emb, item_embO], dim=0)
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            # _, item_embO = torch.split(all_embeddings3, [self.n_users, self.n_items])
            # all_embeddings = torch.cat([user_embO, item_embO], dim=0)

            embeddings_list.append(all_embeddings)

        # 最终嵌入
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)  # + self.headView(epoch)

        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings, item_all_embeddings, alignment_loss


    def forward(self,epoch=500):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        # user_all_embeddings2, item_all_embeddings2 = self.CFChannel()
        # if epoch > 1000:
        #     user_all_embeddings, item_all_embeddings = self.secondStage(user_emb, item_emb)
        # else:
        if epoch<=80:
            users, items, alignment_loss = self.firstStage(user_emb, item_emb, epoch)
        else:
            users, items, alignment_loss = self.secondStage(user_emb, item_emb, epoch)
        # all_embeddings = self.get_ego_embeddings()
        # embeddings_list = [all_embeddings]

        # # LightGCN传播
        # for _ in range(self.n_layers):
        #     all_embeddings =  torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
        #     embeddings_list.append(all_embeddings)
        #
        # # 残差连接
        # final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        # users, items = final_embeddings.split([self.n_users, self.n_items], 0)
        return users, items, alignment_loss
    def CFChannel(self):
        user_emb = self.user_embedding2.weight
        item_emb = self.item_embedding2.weight
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

    def explore_forward(self, users):
        """带探索策略的前向"""
        # 基础嵌入
        base_users, _, _ = self.forward()


        # 动态窗格处理
        pane_mask = self.pane_manager.get_pane_mask(base_users)
        if self.pane_embs is not None:
           base_users = self.llm_enhancer.enhance_user_embeddings(base_users, self.pane_embs,self.pane_manager.pane_labels)

        selected_users = base_users[users]
        explored_users = self.explorer(selected_users, pane_mask[users[:, None], users])

        # 门控融合
        gate = torch.sigmoid(self.gate(
            torch.cat([selected_users, explored_users], dim=1)))
        return gate * selected_users + (1 - gate) * explored_users

    def calculate_loss(self, interaction,epoch):
        Threshold = -1
        # 更新窗格划分
        if epoch > Threshold and epoch % self.config['pane_update_interval'] == 0 and epoch != getattr(self, 'last_pane_update_epoch', -1):
            with torch.no_grad():
                if epoch != getattr(self, 'last_pane_update_epoch', -1):
                    users = self.user_embedding.weight
                else:
                    users = self.user_embedding2.weight
                self.last_pane_update_epoch = epoch
                # self.pane_manager.update_groups(users,self.inter_data)
                pane_labels, pane_members, pane_item_members = self.pane_manager.update_panes(users, self.inter_data)
                _,self.user_patch,self.item_patch = self.pane_manager.generate_group_full_connect_tensor(pane_members, pane_item_members)
                self.user_patch = self.user_patch.to(self.device)
                self.item_patch = self.item_patch.to(self.device)
                # 获取窗格嵌入
                pane_embs = []
                # for pane_id, members in pane_members.items():
                #     # 将列表 B 转换为张量
                #     members = torch.tensor(members).to(self.device)
                #
                #     # 创建布尔掩码
                #     # 使用广播机制比较 A 和 B_tensor 中的每个元素
                #     mask = (self.data.inter_feat['user_id'].unsqueeze(1) == members).any(dim=1)
                #     pane_items = self.data.inter_feat[mask]['item_id']
                #     pane_emb = self.llm_enhancer.get_pane_embedding(pane_items.tolist())
                #     pane_embs.append(pane_emb)
                # self.pane_embs = torch.stack(pane_embs).to(self.config['device'])


        # 基础BPR损失
        user = interaction[self.USER_ID]
        pos = interaction[self.ITEM_ID]
        neg = interaction[self.NEG_ITEM_ID]

        if epoch > Threshold:
            user_emb, item_emb, alignment_loss = self.forward(epoch)
            u_emb = user_emb[user]
            pos_emb = item_emb[pos]
            neg_emb = item_emb[neg]


            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

            # 窗格平滑损失
            pane_centers = self.pane_manager.get_pane_embeddings()
            user_panes = self.pane_manager.pane_labels[user.cpu()]
            center_dist = 1 - F.cosine_similarity(
                u_emb, pane_centers[user_panes])
            pane_loss = center_dist.mean()

            bpr_loss = bpr_loss + 0 * pane_loss - 0 * alignment_loss
        else:
            user_emb, item_emb = self.CFChannel()
            u_emb = user_emb[user]
            pos_emb = item_emb[pos]
            neg_emb = item_emb[neg]

            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

            bpr_loss = bpr_loss

        return bpr_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # u_emb = self.explore_forward(user)
        # i_emb = self.item_embedding(item)
        if self.pane_embs is not None:
            user_emb, item_emb,_ = self.forward()
        else:
            user_emb, item_emb = self.CFChannel()
        u_emb = user_emb[user]
        i_emb = item_emb[item]
        return (u_emb * i_emb).sum(dim=1)