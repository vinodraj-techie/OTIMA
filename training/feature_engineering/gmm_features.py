import pandas as pd
import numpy as np

DATA_PATH = "../../synthetic_data_scripts/data/"
OUTPUT_PATH = "../../synthetic_data_scripts/derived_data/"

def generate_gmm_features():
    sku = pd.read_csv(DATA_PATH + "sku_master.csv")
    demand = pd.read_csv(DATA_PATH + "demand_history.csv")
    picks = pd.read_csv(DATA_PATH + "pick_list.csv")

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
        "weight_kg"
    ]]

    df.to_csv(OUTPUT_PATH + "gmm_features.csv", index=False)
    print("GMM features generated")

if __name__ == "__main__":
    generate_gmm_features()
