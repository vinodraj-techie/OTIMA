import pandas as pd
import numpy as np

DATA_PATH = "../synthetic_data_scripts/derived_data/"

# ----------------------------
# Load data
# ----------------------------
bin_df = pd.read_csv(DATA_PATH + "bin_node_features.csv")
sku_df = pd.read_csv(DATA_PATH + "sku_node_features.csv")
abc_df = pd.read_csv(DATA_PATH + "abc_results.csv")
opt_df = pd.read_csv(DATA_PATH + "optimized_storage.csv")

# ----------------------------
# Merge data
# ----------------------------
df = opt_df.merge(
    bin_df,
    left_on="assigned_bin",
    right_on="bin_id"
).merge(
    abc_df,
    on="sku_id"
)

# ----------------------------
# ABC priority mapping
# ----------------------------
abc_weight = {"A": 3, "B": 2, "C": 1}
df["priority_weight"] = df["abc_class"].map(abc_weight)

# ----------------------------
# Distance calculation
# ----------------------------
df["pick_distance"] = df["distance_from_dispatch"]


# ----------------------------
# SLAP metrics
# ----------------------------
slap_avg_distance = df["pick_distance"].mean()
slap_weighted_distance = (
    (df["pick_distance"] * df["priority_weight"]).sum()
    / df["priority_weight"].sum()
)

# ----------------------------
# RANDOM baseline
# ----------------------------
random_bins = bin_df.sample(n=len(df), replace=True).reset_index(drop=True)
random_distance = np.sqrt(random_bins["x_coord"]**2 + random_bins["y_coord"]**2).mean()

# ----------------------------
# ABC zoning baseline
# ----------------------------
bin_df_sorted = bin_df.sort_values("distance_from_dispatch")

def abc_allocate(abc_class):
    if abc_class == "A":
        return bin_df_sorted.iloc[:int(0.3 * len(bin_df_sorted))]
    elif abc_class == "B":
        return bin_df_sorted.iloc[int(0.3 * len(bin_df_sorted)):int(0.6 * len(bin_df_sorted))]
    else:
        return bin_df_sorted.iloc[int(0.6 * len(bin_df_sorted)):]

abc_bins = []
for cls in df["abc_class"]:
    abc_bins.append(abc_allocate(cls).sample(1))

abc_bins = pd.concat(abc_bins, ignore_index=True)
abc_distance = np.sqrt(abc_bins["x_coord"]**2 + abc_bins["y_coord"]**2).mean()

# ----------------------------
# RESULTS
# ----------------------------
print("\n📊 SLAP EVALUATION RESULTS")
print("---------------------------------")
print(f"Random Allocation Avg Distance : {random_distance:.2f}")
print(f"ABC Allocation Avg Distance    : {abc_distance:.2f}")
print(f"SLAP GNN Avg Distance          : {slap_avg_distance:.2f}")
print(f"SLAP Weighted Distance         : {slap_weighted_distance:.2f}")
