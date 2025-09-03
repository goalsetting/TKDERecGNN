from Model.ffn import *
from torch import nn

from Layer.BGA_layer import BGALayer
import torch




class BGA(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=False, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN(in_channels, hidden_channels)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(
                BGALayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        # self.classifier = nn.Linear(hidden_channels, out_channels)
        self.attn=[]

    def forward(self, x: torch.Tensor, n_users, n_items, patch: torch.Tensor, patch2: torch.Tensor, need_attn=False, user_use=True):
        patch_mask = (patch != -1).float().unsqueeze(-1).long()
        patch_mask2 = (patch2 != -1).float().unsqueeze(-1).long()
        attn_mask = torch.matmul(patch_mask.float(), patch_mask2.float().transpose(1, 2)).int()

        x = self.attribute_encoder(x)
        user, item = torch.split(x, [n_users, n_items])
        for i in range(0, self.layers):
            if user_use:
                x = self.BGALayers[i](user, item, patch,patch2, attn_mask, need_attn)
                if need_attn:
                    self.attn.append(self.BGALayers[i].attn)
            else:
                attn_mask = torch.matmul(patch_mask2.float(), patch_mask.float().transpose(1, 2)).int()
                x = self.BGALayers[i](item, user, patch2,patch, attn_mask, need_attn)
                if need_attn:
                    self.attn.append(self.BGALayers[i].attn)
        # x = self.dropout(x)
        # x = self.classifier(x)
        return x

