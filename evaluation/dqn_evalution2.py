import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.warehouse_env import WarehouseEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10  # ✅ SAME AS TRAINING

# -------------------------
# Model
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Load Environment
# -------------------------
env = WarehouseEnv(
    bin_file="../synthetic_data_scripts/data2/bin_node_features.csv",
    picklist_file="../synthetic_data_scripts/data/pick_list.csv",
    grid_size=10
)

# -------------------------
# Load Model
# -------------------------
STATE_DIM = 7
q_net = DQN(STATE_DIM).to(DEVICE)

model_path = "../models/dqn_route_optimizer2.pt"  # ✅ fixed path
q_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
q_net.eval()

print("✅ Model loaded")

# -------------------------
# Evaluation
# -------------------------
def evaluate_dqn(env, q_net, episodes=100):
    distances = []

    for _ in range(episodes):
        env.reset()
        done = False

        while not done:
            state_actions = env.get_state_action_features()

            # ✅ Apply TOP-K filtering
            idx_sorted = np.argsort(state_actions[:, 4])[:TOP_K]
            filtered_sa = state_actions[idx_sorted]

            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(filtered_sa).to(DEVICE)).squeeze()
                action_local = torch.argmax(q_vals).item()

            action_idx = idx_sorted[action_local]

            _, _, done, _ = env.step(action_idx)

        distances.append(env.total_distance)

    return np.mean(distances)

# -------------------------
# Baselines
# -------------------------
def evaluate_greedy(env, episodes=100):
    return np.mean([env.get_greedy_baseline() for _ in range(episodes)])

def evaluate_random(env, episodes=100):
    distances = []

    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            action = np.random.randint(0, len(env.remaining_bins))
            _, _, done, _ = env.step(action)
        distances.append(env.total_distance)

    return np.mean(distances)

# -------------------------
# Run
# -------------------------
print("\nRunning evaluation...\n")

dqn_dist = evaluate_dqn(env, q_net)
greedy_dist = evaluate_greedy(env)
random_dist = evaluate_random(env)

print(f"DQN Distance     : {dqn_dist:.2f}")
print(f"Greedy Distance  : {greedy_dist:.2f}")
print(f"Random Distance  : {random_dist:.2f}")

# -------------------------
# Comparison
# -------------------------
improvement = (greedy_dist - dqn_dist) / greedy_dist * 100

print("\nPerformance:")
if improvement > 0:
    print(f"✅ DQN beats greedy by {improvement:.2f}%")
else:
    print(f"⚠️ DQN worse than greedy by {abs(improvement):.2f}%")