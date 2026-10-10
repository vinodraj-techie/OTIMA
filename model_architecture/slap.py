import torch
import torch.nn as nn
from torch_geometric.nn import GATConv 
class ImprovedSLAPGNN(nn.Module):
    def __init__(self, sku_dim, bin_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.sku_proj = nn.Sequential(
            nn.Linear(sku_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bin_proj = nn.Sequential(
            nn.Linear(bin_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sku_x, bin_x, edge_index):
        sku_emb = self.sku_proj(sku_x)
        bin_emb = self.bin_proj(bin_x)
        x = torch.cat([sku_emb, bin_emb], dim=0)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_prev = x
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
            if i > 0:
                x = x + x_prev
        
        return x