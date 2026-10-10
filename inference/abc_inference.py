import sys
import pandas as pd
import os

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

demand = pd.read_csv(f"{dataset_path}/demand_history.csv")

sku_demand = demand.groupby("sku_id")["demand_qty"].sum().reset_index()

sku_demand = sku_demand.sort_values("demand_qty", ascending=False)

sku_demand["cum_percent"] = sku_demand["demand_qty"].cumsum() / sku_demand["demand_qty"].sum()

def classify(x):
    if x <= 0.8:
        return "A"
    elif x <= 0.95:
        return "B"
    else:
        return "C"

sku_demand["abc_class"] = sku_demand["cum_percent"].apply(classify)

sku_demand.to_csv(f"{result_path}/abc_results.csv", index=False)

print("ABC inference completed")