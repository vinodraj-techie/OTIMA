import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from collections import deque
from environment.warehouse_env import WarehouseEnv

# -------------------------
# Hyperparameters
# -------------------------
EPISODES = 500
GAMMA = 0.99
LR = 0.0003
BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

TARGET_UPDATE = 10  # Update target network every N episodes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = WarehouseEnv(
    bin_file="../synthetic_data_scripts/derived_data/bin_node_features.csv",
    picklist_file="../synthetic_data_scripts/data/pick_list.csv",
    grid_size=10
)

# -------------------------
# Improved Q-Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output single Q-value
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# State Builder (key improvement)
# -------------------------
def build_state_action_pairs(env):
    """Create state-action pairs for all valid actions"""
    pairs = []
    base_state = np.array([env.current_x, env.current_y], dtype=np.float32)
    
    for bin_id in env.remaining_bins:
        x, y = env.bin_lookup[bin_id]
        # Normalize coordinates
        norm_x = x / 10.0  # Adjust based on your grid_size
        norm_y = y / 10.0
        
        # State: [current_x, current_y, target_x, target_y, distance, num_remaining]
        dist = np.sqrt((env.current_x - x)**2 + (env.current_y - y)**2)
        state_action = np.array([
            env.current_x / 10.0,
            env.current_y / 10.0,
            norm_x,
            norm_y,
            dist / 14.14,  # Normalize by max possible distance (diagonal)
            len(env.remaining_bins) / len(env.pick_df["bin_id"].unique())
        ], dtype=np.float32)
        pairs.append(state_action)
    
    return np.array(pairs)

# -------------------------
# Networks
# -------------------------
STATE_DIM = 7 # Updated state dimension
q_net = DQN(STATE_DIM).to(DEVICE)
target_net = DQN(STATE_DIM).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=MEMORY_SIZE)

# -------------------------
# Environment initialization


# -------------------------
# Training Loop
# -------------------------
# -------------------------
# Training Loop
# -------------------------
epsilon = EPSILON_START
episode_rewards = []
episode_distances = []

print("\nStarting training...")
print(f"Greedy baseline distance: {env.get_greedy_baseline():.2f}\n")

for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get state-action features from environment
        state_actions = env.get_state_action_features()
        
        # Epsilon-greedy selection
        if random.random() < epsilon:
            action_idx = random.randint(0, len(env.remaining_bins) - 1)
        else:
            with torch.no_grad():
                state_actions_tensor = torch.FloatTensor(state_actions).to(DEVICE)
                q_values = q_net(state_actions_tensor).squeeze()
                action_idx = torch.argmax(q_values).item()
        
        # Store current state-action for replay
        current_state_action = state_actions[action_idx]
        
        # Take action
        next_state, reward, done, info = env.step(action_idx)
        total_reward += reward
        
        # Build next state-action pairs for bootstrapping
        if not done:
            next_state_actions = env.get_state_action_features()
        else:
            next_state_actions = None
        
        # Store transition
        memory.append((current_state_action, reward, done, next_state_actions))
        
        # Train
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            
            state_actions_batch = []
            rewards_batch = []
            dones_batch = []
            next_q_values = []
            
            for sa, r, d, next_sa in batch:
                state_actions_batch.append(sa)
                rewards_batch.append(r)
                dones_batch.append(d)
                
                # Calculate max Q-value for next state
                if d or next_sa is None:
                    next_q_values.append(0.0)
                else:
                    with torch.no_grad():
                        next_sa_tensor = torch.from_numpy(np.array(next_sa)).float().to(DEVICE)
                        max_next_q = target_net(next_sa_tensor).max().item()
                        next_q_values.append(max_next_q)
            
            # Convert to tensors efficiently using numpy first
            state_actions_batch = torch.from_numpy(np.array(state_actions_batch)).float().to(DEVICE)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=DEVICE)
            dones_batch = torch.tensor(dones_batch, dtype=torch.float32, device=DEVICE)
            next_q_values = torch.tensor(next_q_values, dtype=torch.float32, device=DEVICE)
            
            # Current Q-values
            current_q = q_net(state_actions_batch).squeeze()
            
            # Target Q-values
            target_q = rewards_batch + GAMMA * next_q_values * (1 - dones_batch)
            
            loss = loss_fn(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
    
    # Update target network
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())
    
    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    episode_rewards.append(total_reward)
    episode_distances.append(env.total_distance)
    
    if ep % 20 == 0:
        avg_dist = np.mean(episode_distances[-20:]) if len(episode_distances) >= 20 else env.total_distance
        avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else total_reward
        print(f"Ep {ep:3d} | Dist: {env.total_distance:7.2f} | Avg Dist: {avg_dist:7.2f} | Avg Reward: {avg_reward:7.2f} | ε: {epsilon:.3f}")
    # Update target network
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())
    
    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    episode_rewards.append(total_reward)
    episode_distances.append(env.total_distance)
    
    if ep % 20 == 0:
        avg_dist = np.mean(episode_distances[-20:]) if len(episode_distances) >= 20 else env.total_distance
        print(f"Ep {ep:3d} | Dist: {env.total_distance:7.2f} | Avg: {avg_dist:7.2f} | ε: {epsilon:.3f}")

print(f"\n✅ Training complete. Final avg distance: {np.mean(episode_distances[-50:]):.2f}")
torch.save(q_net.state_dict(), "../models/dqn_route_optimizer.pt")