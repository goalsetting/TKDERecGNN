# %% ########################
# 模块2：探索策略模块 modules/exploration.py
#############################
import torch.nn as nn


class TransformerExplorer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=2 * dim),
            num_layers=2  # 指定层数
        )
        self.explore_proj = nn.Linear(dim, dim)

    def forward(self, x, pane_mask):
        """
        x: 节点嵌入 [N, D]
        pane_mask: 窗格掩码 [N, N]
        """
        # 调整输入维度为 (S=1024, N=1, D)
        x_seq = x.unsqueeze(0)  # [1, 1024, D]
        x_seq = x_seq.permute(1, 0, 2)  # [1024, 1, D] → (S=1024, N=1, D)

        # 创建注意力掩码（形状[S=1024, S=1024]）
        attn_mask = ~pane_mask  # [1024, 1024]
        attn_mask = attn_mask

        # 使用TransformerEncoder（假设self.transformer是nn.TransformerEncoder）
        explored = self.transformer(
            x_seq,
            mask=attn_mask  # 确保mask形状为(S, S)
        ).permute(1, 0, 2).squeeze(0)  # 转换回[N, D]

        # 残差连接
        return x + self.explore_proj(explored)