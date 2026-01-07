import pandas as pd
import numpy as np
from config import *

sku_df = pd.read_csv("data/sku_master.csv")
inv_df = pd.read_csv("data/inventory_status.csv")

orders = []
pick_rows = []

for order_id in range(NUM_ORDERS):
    order_code = f"ORD_{order_id:07d}"
    num_lines = np.random.randint(1, MAX_ORDER_LINES + 1)
    skus = sku_df.sample(num_lines)

    orders.append({
        "order_id": order_code,
        "order_priority": np.random.choice(["low", "medium", "high"], p=[0.5, 0.3, 0.2])
    })

    for _, sku in skus.iterrows():
        bin_id = inv_df[inv_df["sku_id"] == sku["sku_id"]]["bin_id"].values[0]

        pick_rows.append({
            "order_id": order_code,
            "sku_id": sku["sku_id"],
            "bin_id": bin_id,
            "pick_qty": np.random.randint(1, 20)
        })

pd.DataFrame(orders).to_csv("data/orders.csv", index=False)
pd.DataFrame(pick_rows).to_csv("data/pick_list.csv", index=False)

print("Orders & pick list generated")
