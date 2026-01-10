import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

# Load base datasets
sku_df = pd.read_csv("data/sku_master.csv")
layout_df = pd.read_csv("data/warehouse_layout.csv")
inv_df = pd.read_csv("data/inventory_status.csv")
pick_df = pd.read_csv("data/pick_list.csv")

# ===============================
# 1️⃣ SKU NODE FEATURES
# ===============================
sku_features = sku_df[[
    "sku_id",
    "unit_cost",
    "annual_demand",
    "weight_kg",
    "volume_cu_m"
]].copy()

sku_features["pick_frequency"] = (
    pick_df.groupby("sku_id").size().reindex(sku_features["sku_id"]).fillna(0).values
)

sku_features.to_csv("derived_data/sku_node_features.csv", index=False)

# ===============================
# 2️⃣ BIN NODE FEATURES
# ===============================
bin_features = layout_df.copy()
bin_features["distance_from_dispatch"] = np.sqrt(
    bin_features["x_coord"]**2 + bin_features["y_coord"]**2
)

bin_features.to_csv("derived_data/bin_node_features.csv", index=False)

# ===============================
# 3️⃣ SKU → BIN EDGES
# ===============================
sku_bin_edges = inv_df[["sku_id", "bin_id"]].copy()
sku_bin_edges["edge_type"] = "stored_in"

sku_bin_edges.to_csv("derived_data/sku_bin_edges.csv", index=False)

# ===============================
# 4️⃣ BIN → BIN ADJACENCY EDGES
# ===============================
edges = []

for _, b1 in layout_df.iterrows():
    neighbors = layout_df[
        (abs(layout_df["x_coord"] - b1["x_coord"]) <= 1) &
        (abs(layout_df["y_coord"] - b1["y_coord"]) <= 1) &
        (layout_df["bin_id"] != b1["bin_id"])
    ]

    for _, b2 in neighbors.iterrows():
        edges.append({
            "from_bin": b1["bin_id"],
            "to_bin": b2["bin_id"],
            "distance": np.linalg.norm(
                [b1["x_coord"] - b2["x_coord"], b1["y_coord"] - b2["y_coord"]]
            )
        })

pd.DataFrame(edges).to_csv("derived_data/bin_adjacency_edges.csv", index=False)

# ===============================
# 5️⃣ SKU → SKU CO-PICK EDGES
# ===============================
copick_counter = Counter()

for order_id, group in pick_df.groupby("order_id"):
    skus = group["sku_id"].tolist()
    for pair in combinations(sorted(skus), 2):
        copick_counter[pair] += 1

copick_edges = [{
    "sku_1": k[0],
    "sku_2": k[1],
    "co_pick_frequency": v
} for k, v in copick_counter.items()]

pd.DataFrame(copick_edges).to_csv("derived_data/sku_sku_copick_edges.csv", index=False)

print("SLAP GNN graph dataset generated successfully")
