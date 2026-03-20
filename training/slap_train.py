import torch
import pandas as pd
import numpy as np
import os

from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

DATA_PATH = "../synthetic_data_scripts/derived_data/"
MODEL_PATH = "../models/slap/"
os.makedirs(MODEL_PATH, exist_ok=True)

# -------------------------
# Load data
# -------------------------
sku_df = pd.read_csv(DATA_PATH + "sku_node_features_scaled.csv")
bin_df = pd.read_csv(DATA_PATH + "bin_node_features_scaled.csv")
edges_df = pd.read_csv(DATA_PATH + "sku_bin_edges.csv")

# -------------------------
# Precompute near and far bins (distance-aware)
# -------------------------
bin_distances = bin_df["distance_from_dispatch"].values
near_bins = np.argsort(bin_distances)[: int(0.3 * len(bin_distances))]
far_bins = np.argsort(bin_distances)[-int(0.3 * len(bin_distances)):]

# -------------------------
# Node features (SEPARATE)
# -------------------------
sku_numeric_df = sku_df.drop(columns=["sku_id"]).select_dtypes(include=[np.number])
bin_numeric_df = bin_df.drop(columns=["bin_id"]).select_dtypes(include=[np.number])

sku_x = torch.tensor(sku_numeric_df.values, dtype=torch.float)
bin_x = torch.tensor(bin_numeric_df.values, dtype=torch.float)

num_skus = sku_x.shape[0]
num_bins = bin_x.shape[0]

# -------------------------
# Encode IDs
# -------------------------
sku_map = dict(zip(sku_df["sku_id"], range(num_skus)))
bin_map = dict(zip(bin_df["bin_id"], range(num_bins)))

edges_df["sku_idx"] = edges_df["sku_id"].map(sku_map)
edges_df["bin_idx"] = edges_df["bin_id"].map(bin_map) + num_skus

# -------------------------
# Train / Val / Test split
# -------------------------
train_edges, temp_edges = train_test_split(
    edges_df, test_size=0.3, random_state=42
)
val_edges, test_edges = train_test_split(
    temp_edges, test_size=0.5, random_state=42
)

def build_edge_index(df):
    edges = []
    for _, r in df.iterrows():
        edges.append([r["sku_idx"], r["bin_idx"]])
        edges.append([r["bin_idx"], r["sku_idx"]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

edge_index = build_edge_index(train_edges)

# -------------------------
# GNN Model
# -------------------------
class SLAPGNN(nn.Module):
    def __init__(self, sku_dim, bin_dim, hidden_dim=64):
        super().__init__()
        self.sku_proj = nn.Linear(sku_dim, hidden_dim)
        self.bin_proj = nn.Linear(bin_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, sku_x, bin_x, edge_index):
        sku_emb = self.sku_proj(sku_x)
        bin_emb = self.bin_proj(bin_x)
        x = torch.cat([sku_emb, bin_emb], dim=0)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = SLAPGNN(
    sku_dim=sku_x.shape[1],
    bin_dim=bin_x.shape[1]
)

optimizer = Adam(model.parameters(), lr=0.01)

# -------------------------
# Training (DISTANCE-AWARE CONTRASTIVE LOSS)
# -------------------------
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    emb = model(sku_x, bin_x, edge_index)

    loss = 0.0
    for i in range(num_skus):
        sku_emb = emb[i]

        good_bin_idx = num_skus + np.random.choice(near_bins)
        bad_bin_idx = num_skus + np.random.choice(far_bins)

        good_bin_emb = emb[good_bin_idx]
        bad_bin_emb = emb[bad_bin_idx]

        loss += torch.norm(sku_emb - good_bin_emb) ** 2
        loss -= torch.norm(sku_emb - bad_bin_emb) ** 2

    loss = loss / num_skus
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_PATH + "slap_gnn.pt")

# -------------------------
# Inference
# -------------------------
model.eval()
emb = model(sku_x, bin_x, edge_index).detach()

assignments = []
for i in range(num_skus):
    dists = torch.norm(emb[i] - emb[num_skus:], dim=1)
    best_bin = bin_df.iloc[dists.argmin().item()]["bin_id"]
    assignments.append(best_bin)

out = pd.DataFrame({
    "sku_id": sku_df["sku_id"],
    "assigned_bin": assignments
})

out.to_csv(DATA_PATH + "optimized_storage.csv", index=False)

print("✅ SLAP training & inference completed successfully")
