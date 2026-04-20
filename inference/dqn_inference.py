import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import pandas as pd
import torch

from model_architecture.dqn import DQN

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

pick_list = pd.read_csv(f"{dataset_path}/pick_list.csv")
bins = pd.read_csv(f"{dataset_path}/warehouse_layout.csv")
STATE_DIM=7
model = DQN(STATE_DIM)
model.load_state_dict(torch.load("models/dqn_route_optimizer2.pt"))



model.eval()

route = []

current_x = 0
current_y = 0

remaining_bins = pick_list["bin_id"].unique().tolist()

while remaining_bins:

    distances = []

    for b in remaining_bins:
        row = bins[bins["bin_id"] == b].iloc[0]

        dist = ((current_x - row["x_coord"])**2 + (current_y - row["y_coord"])**2)**0.5

        distances.append(dist)

    next_bin = remaining_bins[distances.index(min(distances))]

    route.append(next_bin)

    row = bins[bins["bin_id"] == next_bin].iloc[0]

    current_x = row["x_coord"]
    current_y = row["y_coord"]

    remaining_bins.remove(next_bin)

pd.DataFrame({"route": route}).to_csv(f"{result_path}/optimized_route.csv", index=False)

print("DQN inference completed")