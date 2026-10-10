import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import *

sku_df = pd.read_csv("data2/sku_master.csv")
start_date = datetime(2024, 1, 1)

rows = []

for _, sku in sku_df.iterrows():
    base = sku["annual_demand"] / NUM_DAYS

    for day in range(NUM_DAYS):
        date = start_date + timedelta(days=day)
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * day / 365)
        noise = np.random.normal(0, base * 0.2)
        demand = max(0, int(base * seasonal + noise))

        rows.append({
            "sku_id": sku["sku_id"],
            "date": date,
            "demand_qty": demand,
            "promotion_flag": np.random.choice([0, 1], p=[0.9, 0.1])
        })

df = pd.DataFrame(rows)
df.to_csv("data2/demand_history.csv", index=False)
print("Demand history generated")
