import pandas as pd
import numpy as np

sku_df = pd.read_csv("data/sku_master.csv")
layout_df = pd.read_csv("data/warehouse_layout.csv")

bins = layout_df["bin_id"].sample(len(sku_df)).values

df = pd.DataFrame({
    "sku_id": sku_df["sku_id"],
    "bin_id": bins,
    "on_hand_qty": np.random.randint(50, 5000, len(sku_df)),
    "reorder_point": np.random.randint(100, 1000, len(sku_df)),
    "safety_stock": np.random.randint(50, 500, len(sku_df))
})

df.to_csv("data/inventory_status.csv", index=False)
print("Inventory status generated")
