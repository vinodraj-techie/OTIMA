import pandas as pd
import numpy as np
from config import *

sku_ids = [f"SKU_{i:05d}" for i in range(NUM_SKUS)]

data = {
    "sku_id": sku_ids,
    "unit_cost": np.round(np.random.uniform(5, 500, NUM_SKUS), 2),
    "annual_demand": np.random.randint(100, 50000, NUM_SKUS),
    "weight_kg": np.round(np.random.uniform(0.1, 30, NUM_SKUS), 2),
    "volume_cu_m": np.round(np.random.uniform(0.001, 0.2, NUM_SKUS), 3),
    "handling_type": np.random.choice(["normal", "fragile", "hazardous"], NUM_SKUS),
}

df = pd.DataFrame(data)

# ABC Classification
df["annual_consumption_value"] = df["unit_cost"] * df["annual_demand"]
df = df.sort_values("annual_consumption_value", ascending=False)

cumulative = df["annual_consumption_value"].cumsum() / df["annual_consumption_value"].sum()

df["abc_class"] = np.where(
    cumulative <= 0.7, "A",
    np.where(cumulative <= 0.9, "B", "C")
)

df.to_csv("data/sku_master.csv", index=False)
print("SKU master generated")
