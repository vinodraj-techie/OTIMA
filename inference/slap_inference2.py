import sys
import os
import json
import pandas as pd
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from model_architecture.slap import ImprovedSLAPGNN

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

# -----------------------
# Load metadata (🔥 FIX)
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "slap_fixed", "slap_gnn_best.pt")
meta_path = os.path.join(BASE_DIR, "models", "slap_fixed", "model_metadata.json")

with open(meta_path, "r") as f:
    metadata = json.load(f)

sku_cols = metadata["sku_feature_columns"]
bin_cols = metadata["bin_feature_columns"]

print("Using SKU features:", sku_cols)
print("Using BIN features:", bin_cols)

# -----------------------
# Load scaled features
# -----------------------
sku_df = pd.read_csv(os.path.join(dataset_path, "sku_node_features_scaled.csv"))
bin_df = pd.read_csv(os.path.join(dataset_path, "bin_node_features_scaled.csv"))

# -----------------------
# Use EXACT SAME FEATURES
# -----------------------
sku_numeric = sku_df[sku_cols]
bin_numeric = bin_df[bin_cols]

# Safety check
if sku_numeric.shape[1] != len(sku_cols):
    raise ValueError("❌ SKU feature mismatch with training")

if bin_numeric.shape[1] != len(bin_cols):
    raise ValueError("❌ BIN feature mismatch with training")

sku_tensor = torch.tensor(sku_numeric.values, dtype=torch.float32)
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

print("Loading model from:", model_path)

checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -----------------------
# Dummy edge index
# -----------------------
num_skus = sku_tensor.shape[0]
num_bins = bin_tensor.shape[0]

edge_index = torch.empty((2, 0), dtype=torch.long)

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

print("✅ SLAP inference completed successfully")