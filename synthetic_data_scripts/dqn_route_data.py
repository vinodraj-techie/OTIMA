import pandas as pd
import numpy as np

# Load datasets
layout_df = pd.read_csv("data2/warehouse_layout.csv")
pick_df = pd.read_csv("data2/pick_list.csv")

# Merge coordinates
pick_df = pick_df.merge(
    layout_df[["bin_id", "x_coord", "y_coord"]],
    on="bin_id",
    how="left"
)

episodes = []

for order_id, group in pick_df.groupby("order_id"):
    bins = group[["bin_id", "x_coord", "y_coord"]].drop_duplicates().values.tolist()

    for i in range(len(bins)):
        current = bins[i]
        for j in range(len(bins)):
            if i == j:
                continue

            next_bin = bins[j]
            distance = np.linalg.norm(
                [current[1] - next_bin[1], current[2] - next_bin[2]]
            )

            episodes.append({
                "order_id": order_id,
                "current_bin": current[0],
                "current_x": current[1],
                "current_y": current[2],
                "next_bin": next_bin[0],
                "next_x": next_bin[1],
                "next_y": next_bin[2],
                "reward": -distance
            })

pd.DataFrame(episodes).to_csv("data2/dqn_episodes.csv", index=False)

print("DQN routing dataset generated successfully")
