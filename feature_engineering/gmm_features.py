import pandas as pd
import sys
import os

def generate_gmm_features(input_path, output_path):

    sku = pd.read_csv(os.path.join(input_path, "sku_master.csv"))
    demand = pd.read_csv(os.path.join(input_path, "demand_history.csv"))
    picks = pd.read_csv(os.path.join(input_path, "pick_list.csv"))

    demand_stats = demand.groupby("sku_id")["demand_qty"].agg(
        avg_daily_demand="mean",
        demand_std="std"
    ).reset_index()

    pick_freq = picks.groupby("sku_id").size().reset_index(name="pick_frequency")

    df = sku.merge(demand_stats, on="sku_id")
    df = df.merge(pick_freq, on="sku_id", how="left")
    df["pick_frequency"] = df["pick_frequency"].fillna(0)

    df = df[[
        "sku_id",
        "avg_daily_demand",
        "demand_std",
        "pick_frequency",
        "volume_cu_m",
        
    ]]

    os.makedirs(output_path, exist_ok=True)

    df.to_csv(os.path.join(output_path, "gmm_features.csv"), index=False)

    print("✅ GMM features generated")


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_gmm_features(input_path, output_path)