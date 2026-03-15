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
    # Load GMM and ABC results
    # -------------------------------
    gmm_clusters = pd.read_csv(os.path.join(output_path, "gmm_features.csv"))
    abc_results = pd.read_csv(os.path.join(output_path, "abc_results.csv"))

    # -------------------------------
    # Merge GMM clusters into SKU nodes
    # -------------------------------
    sku_nodes = sku_nodes.merge(
        gmm_clusters,
        on="sku_id",
        how="left"
    )

    # -------------------------------
    # Merge ABC class into SKU nodes
    # -------------------------------
    sku_nodes = sku_nodes.merge(
        abc_results[["sku_id", "abc_class"]],
        on="sku_id",
        how="left"
    )

    # -------------------------------
    # Encode ABC class as numeric priority
    # -------------------------------
    abc_map = {"A": 3, "B": 2, "C": 1}
    sku_nodes["abc_priority"] = sku_nodes["abc_class"].map(abc_map)

    # -------------------------------
    # SAFE normalization for SKU features
    # -------------------------------
    sku_numeric_cols = [
        "unit_cost",
        "annual_demand",
        "weight_kg",
        "volume_cu_m",
        "pick_frequency",
        "gmm_cluster",
        "abc_priority"
    ]

    for col in sku_numeric_cols:
        if col in sku_nodes.columns:
            std = sku_nodes[col].std()
            if std and std > 0:
                sku_nodes[col] = (sku_nodes[col] - sku_nodes[col].mean()) / std

    # -------------------------------
    # SAFE normalization for bin features
    # -------------------------------
    bin_numeric_cols = [
        "x_coord",
        "y_coord",
        "distance_from_dispatch"
    ]

    for col in bin_numeric_cols:
        if col in bin_nodes.columns:
            std = bin_nodes[col].std()
            if std and std > 0:
                bin_nodes[col] = (bin_nodes[col] - bin_nodes[col].mean()) / std

    # -------------------------------
    # Create output directory
    # -------------------------------
    os.makedirs(output_path, exist_ok=True)

    # -------------------------------
    # Save scaled features
    # -------------------------------
    sku_nodes.to_csv(
        os.path.join(output_path, "sku_node_features_scaled.csv"),
        index=False
    )

    bin_nodes.to_csv(
        os.path.join(output_path, "bin_node_features_scaled.csv"),
        index=False
    )

    print("✅ SLAP GNN features prepared with GMM clustering and ABC priority")


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_slap_features(input_path, output_path)