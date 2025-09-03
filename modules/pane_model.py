# %% ########################
# 模块3：窗格感知模型 modules/pane_model.py
#############################
import torch
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.layers import MLPLayers
import torch.nn as nn


class PaneAwareModel(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # 基础嵌入层
        self.user_embedding = nn.Embedding(self.n_users, config['embedding_size'])
        self.item_embedding = nn.Embedding(self.n_items, config['embedding_size'])

        # 窗格增强模块
        self.enhance_proj = MLPLayers(
            [config['embedding_size'] * 2, config['embedding_size']],
            activation='gelu'
        )

        # 上下文门控
        self.gate = nn.Linear(config['embedding_size'] * 2, 1)

    def forward(self, user, item, pane_emb=None):
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)

        if pane_emb is not None:
            # 拼接窗格特征
            enhanced_u = torch.cat([u_emb, pane_emb], dim=1)
            enhanced_u = self.enhance_proj(enhanced_u)

            # 门控融合
            gate = torch.sigmoid(self.gate(torch.cat([u_emb, enhanced_u], dim=1)))
            u_emb = gate * u_emb + (1 - gate) * enhanced_u

        return torch.mul(u_emb, i_emb).sum(dim=1)

    def calculate_loss(self, interaction, pane_emb=None):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_score = self.forward(user, pos_item, pane_emb)
        neg_score = self.forward(user, neg_item, pane_emb)

        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
        return loss