import pandas as pd
from config import *

rows = []
bin_id = 0

zones = ["A", "B", "C"]

for zone in zones:
    for aisle in range(AISLES_PER_ZONE):
        for rack in range(RACKS_PER_AISLE):
            for bin_ in range(BINS_PER_RACK):
                rows.append({
                    "bin_id": f"BIN_{bin_id:06d}",
                    "zone": zone,
                    "aisle": aisle,
                    "rack": rack,
                    "bin": bin_,
                    "x_coord": aisle * 10 + rack,
                    "y_coord": bin_
                })
                bin_id += 1

df = pd.DataFrame(rows)
df.to_csv("data2/warehouse_layout.csv", index=False)
print("Warehouse layout generated")
