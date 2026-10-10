import pandas as pd
import numpy as np
import os
import sys

def generate_slap_features(input_path, output_path):

    # -------------------------------
    # Load base SLAP node features
    # -------------------------------
    sku_nodes = pd.read_csv(os.path.join(input_path, "sku_node_features.csv"))
    bin_nodes = pd.read_csv(os.path.join(input_path, "bin_node_features.csv"))

    # -------------------------------
    # Load GMM + ABC
    # -------------------------------
    gmm_features = pd.read_csv(os.path.join(output_path, "gmm_features.csv"))
    abc_results = pd.read_csv(os.path.join(output_path, "abc_results.csv"))

    print("GMM columns:", gmm_features.columns.tolist())

    # -------------------------------
    # Merge GMM features
    # -------------------------------
    sku_nodes = sku_nodes.merge(gmm_features, on="sku_id", how="left")

    # -------------------------------
    # Merge ABC
    # -------------------------------
    sku_nodes = sku_nodes.merge(
        abc_results[["sku_id", "abc_class"]],
        on="sku_id",
        how="left"
    )

# -------------------------------
# FIX: HANDLE COLUMN COLLISIONS PROPERLY
# -------------------------------

    for col in ["volume_cu_m", "pick_frequency", "weight_kg"]:
        col_x = f"{col}_x"
        col_y = f"{col}_y"

        if col_x in sku_nodes.columns and col_y in sku_nodes.columns:
           sku_nodes[col] = sku_nodes[col_x].combine_first(sku_nodes[col_y])
           sku_nodes = sku_nodes.drop(columns=[col_x, col_y])

        elif col_y in sku_nodes.columns:
           sku_nodes = sku_nodes.rename(columns={col_y: col})

        elif col_x in sku_nodes.columns:
           sku_nodes = sku_nodes.rename(columns={col_x: col})

    # -------------------------------
    # FIX: Handle missing columns
    # -------------------------------
    if "weight_kg" not in sku_nodes.columns:
        print("⚠️ Missing weight_kg, filling with 0")
        sku_nodes["weight_kg"] = 0.0

    # -------------------------------
    # Encode ABC
    # -------------------------------
    abc_map = {"A": 3, "B": 2, "C": 1}
    sku_nodes["abc_priority"] = sku_nodes["abc_class"].map(abc_map)

    # -------------------------------
    # FINAL FEATURE LIST (🔥 IMPORTANT)
    # -------------------------------
    final_sku_cols = [
        "sku_id",
        "unit_cost",
        "annual_demand",
        "weight_kg",
        "volume_cu_m",
        "pick_frequency",
        "avg_daily_demand",
        "demand_std",
        "abc_priority"
    ]

    # Validate columns
    missing = [col for col in final_sku_cols if col not in sku_nodes.columns]
    if missing:
        raise ValueError(f"❌ Missing columns: {missing}")

    # Keep only required columns
    sku_nodes = sku_nodes[final_sku_cols]

    # -------------------------------
    # Normalize SKU features
    # -------------------------------
    for col in final_sku_cols:
        if col != "sku_id":
            std = sku_nodes[col].std()
            if std and std > 0:
                sku_nodes[col] = (sku_nodes[col] - sku_nodes[col].mean()) / std

    # -------------------------------
    # LOCK BIN FEATURES
    # -------------------------------
    final_bin_cols = [
        "bin_id",
        "x_coord",
        "y_coord",
        "distance_from_dispatch"
    ]

    bin_nodes = bin_nodes[final_bin_cols]

    # Normalize bin features
    for col in final_bin_cols:
        if col != "bin_id":
            std = bin_nodes[col].std()
            if std and std > 0:
                bin_nodes[col] = (bin_nodes[col] - bin_nodes[col].mean()) / std

    # -------------------------------
    # Save outputs
    # -------------------------------
    os.makedirs(output_path, exist_ok=True)

    sku_nodes.to_csv(
        os.path.join(output_path, "sku_node_features_scaled.csv"),
        index=False
    )

    bin_nodes.to_csv(
        os.path.join(output_path, "bin_node_features_scaled.csv"),
        index=False
    )

    print("✅ SLAP GNN features prepared (clean & stable)")


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_slap_features(input_path, output_path)