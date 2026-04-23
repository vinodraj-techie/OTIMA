import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import pandas as pd
import torch
import numpy as np

from model_architecture.slap import ImprovedSLAPGNN

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

# -----------------------
# Load scaled features
# -----------------------
sku_df = pd.read_csv(os.path.join(dataset_path, "sku_node_features_scaled.csv"))

# Encode abc_class exactly like training
if "abc_class" in sku_df.columns:
    sku_df["abc_class"] = sku_df["abc_class"].map({
        "A": 0,
        "B": 1,
        "C": 2
    })

sku_numeric = sku_df.drop(columns=["sku_id"]).select_dtypes(include=[np.number])

sku_tensor = torch.tensor(
    sku_numeric.values,
    dtype=torch.float32
)
bin_df = pd.read_csv(os.path.join(dataset_path, "bin_node_features_scaled.csv"))


bin_numeric = bin_df.drop(columns=["bin_id"]).select_dtypes(include=[np.number])


bin_tensor = torch.tensor(bin_numeric.values, dtype=torch.float32)




# -----------------------
# Model dimensions
# -----------------------
sku_dim = sku_tensor.shape[1]
bin_dim = bin_tensor.shape[1]

# -----------------------
# Load trained model
# -----------------------
model = ImprovedSLAPGNN(
    sku_dim=sku_dim,
    bin_dim=bin_dim
)

checkpoint = torch.load("models/slap_fixed/slap_gnn_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------
# Dummy edge index
# -----------------------
num_skus = sku_tensor.shape[0]
num_bins = bin_tensor.shape[0]

edge_index = torch.empty((2,0), dtype=torch.long)

# -----------------------
# Run inference
# -----------------------
with torch.no_grad():

    emb = model(sku_tensor, bin_tensor, edge_index)

    sku_emb = emb[:num_skus]
    bin_emb = emb[num_skus:]

    distances = torch.cdist(sku_emb, bin_emb, p=2)

    best_bin_indices = torch.argmin(distances, dim=1).numpy()

# -----------------------
# Save assignments
# -----------------------
assignments = [bin_df.iloc[idx]["bin_id"] for idx in best_bin_indices]

results = pd.DataFrame({
    "sku_id": sku_df["sku_id"],
    "assigned_bin": assignments
})

results.to_csv(
    os.path.join(result_path, "optimized_storage.csv"),
    index=False
)

print("SLAP inference completed")





