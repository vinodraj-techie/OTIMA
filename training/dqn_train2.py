import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.warehouse_env2 import WarehouseEnv

# -------------------------
# Hyperparameters
# -------------------------
EPISODES = 500
GAMMA = 0.99
LR = 0.0003
BATCH_SIZE = 64
MEMORY_SIZE = 20000

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998

TARGET_UPDATE = 10
TOP_K = 10  # ✅ Action filtering

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Environment
# -------------------------
env = WarehouseEnv(
    bin_file="../synthetic_data_scripts/data/bin_node_features.csv",
    picklist_file="../synthetic_data_scripts/data/pick_list.csv",
    grid_size=10
)

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

STATE_DIM = 7  # ✅ Correct now
q_net = DQN(STATE_DIM).to(DEVICE)
target_net = DQN(STATE_DIM).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

# -------------------------
# Training
# -------------------------
epsilon = EPSILON_START

for ep in range(EPISODES):
    env.reset()
    done = False
    total_reward = 0

    while not done:
        state_actions = env.get_state_action_features()

        # ✅ Filter nearest actions
        idx_sorted = np.argsort(state_actions[:, 4])[:TOP_K]
        state_actions = state_actions[idx_sorted]

        if random.random() < epsilon:
            action_local = random.randint(0, len(state_actions) - 1)
        else:
            with torch.no_grad():
                q_vals = q_net(torch.FloatTensor(state_actions).to(DEVICE)).squeeze()
                action_local = torch.argmax(q_vals).item()

        action_idx = idx_sorted[action_local]

        sa = state_actions[action_local]

        _, reward, done, _ = env.step(action_idx)
        total_reward += reward

        next_sa = env.get_state_action_features() if not done else None

        memory.append((sa, reward, done, next_sa))

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)

            sa_batch, r_batch, d_batch, next_q = [], [], [], []

            for sa, r, d, nsa in batch:
                sa_batch.append(sa)
                r_batch.append(r)
                d_batch.append(d)

                if d or nsa is None:
                    next_q.append(0)
                else:
                    with torch.no_grad():
                        nsa = torch.FloatTensor(nsa).to(DEVICE)
                        next_q.append(target_net(nsa).max().item())

            sa_batch = torch.FloatTensor(sa_batch).to(DEVICE)
            r_batch = torch.FloatTensor(r_batch).to(DEVICE)
            d_batch = torch.FloatTensor(d_batch).to(DEVICE)
            next_q = torch.FloatTensor(next_q).to(DEVICE)

            q = q_net(sa_batch).squeeze()
            target = r_batch + GAMMA * next_q * (1 - d_batch)

            loss = nn.MSELoss()(q, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

    # ✅ Update target network
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if ep % 20 == 0:
        print(f"Ep {ep} | Dist: {env.total_distance:.2f} | ε: {epsilon:.3f}")

print("✅ Training Complete")
torch.save(q_net.state_dict(), "../models/dqn_route_optimizer2.pt")