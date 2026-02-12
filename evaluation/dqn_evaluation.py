import numpy as np
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environment.warehouse_env import WarehouseEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Q-Network (same as training)
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Load Environment
# -------------------------
env = WarehouseEnv(
    bin_file="../synthetic_data_scripts/derived_data/bin_node_features.csv",
    picklist_file="../synthetic_data_scripts/data/pick_list.csv",
    grid_size=10
)

print("=" * 70)
print("DQN ROUTE OPTIMIZER - EVALUATION")
print("=" * 70)
print(f"Number of bins to visit: {env.num_bins}")
print(f"Grid size: {env.grid_size}x{env.grid_size}")

# -------------------------
# Load Model
# -------------------------
STATE_DIM = 7
q_net = DQN(STATE_DIM).to(DEVICE)

model_path = "../models/dqn/dqn_route_optimizer.pt"
if not os.path.exists(model_path):
    print(f"\n❌ Model not found at {model_path}")
    print("Please train the model first using dqn_train.py")
    sys.exit(1)

checkpoint = torch.load(model_path, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    q_net.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✅ Model loaded from {model_path}")
    if 'greedy_baseline' in checkpoint:
        print(f"Training greedy baseline: {checkpoint['greedy_baseline']:.2f}")
else:
    q_net.load_state_dict(checkpoint)
    print(f"\n✅ Model loaded from {model_path}")

q_net.eval()

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_dqn(env, q_net, num_episodes=100, verbose=False):
    """
    Evaluate the DQN policy over multiple episodes.
    
    Args:
        env: WarehouseEnv instance
        q_net: Trained DQN model
        num_episodes: Number of evaluation episodes
        verbose: Print episode-by-episode results
    
    Returns:
        dict with evaluation metrics
    """
    distances = []
    rewards = []
    routes = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        route = [("Dispatch", 0.0, 0.0)]
        
        while not done:
            # Get state-action features
            state_actions = env.get_state_action_features()
            
            # Greedy action selection (no exploration)
            with torch.no_grad():
                state_actions_tensor = torch.FloatTensor(state_actions).to(DEVICE)
                q_values = q_net(state_actions_tensor).squeeze()
                action_idx = torch.argmax(q_values).item()
            
            # Record the bin we're visiting
            bin_id = env.remaining_bins[action_idx]
            x, y = env.bin_lookup[bin_id]
            route.append((bin_id, x, y))
            
            # Take action
            next_state, reward, done, info = env.step(action_idx)
            total_reward += reward
        
        distances.append(env.total_distance)
        rewards.append(total_reward)
        routes.append(route)
        
        if verbose:
            print(f"Episode {ep+1:3d} | Distance: {env.total_distance:7.2f} | Reward: {total_reward:7.2f}")
    
    return {
        'distances': distances,
        'rewards': rewards,
        'routes': routes,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'mean_reward': np.mean(rewards),
    }

# -------------------------
# Greedy Baseline
# -------------------------
def evaluate_greedy(env, num_episodes=100):
    """Evaluate greedy nearest-neighbor baseline"""
    distances = []
    
    for _ in range(num_episodes):
        dist = env.get_greedy_baseline()
        distances.append(dist)
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
    }

# -------------------------
# Random Baseline
# -------------------------
def evaluate_random(env, num_episodes=100):
    """Evaluate random action selection baseline"""
    distances = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Random action
            action_idx = np.random.randint(0, len(env.remaining_bins))
            next_state, reward, done, info = env.step(action_idx)
        
        distances.append(env.total_distance)
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
    }

# -------------------------
# Run Evaluation
# -------------------------
print("\n" + "-" * 70)
print("Running evaluation...")
print("-" * 70)

NUM_EVAL_EPISODES = 100

print(f"\nEvaluating DQN policy ({NUM_EVAL_EPISODES} episodes)...")
dqn_results = evaluate_dqn(env, q_net, num_episodes=NUM_EVAL_EPISODES, verbose=False)

print(f"\nEvaluating Greedy baseline ({NUM_EVAL_EPISODES} episodes)...")
greedy_results = evaluate_greedy(env, num_episodes=NUM_EVAL_EPISODES)

print(f"\nEvaluating Random baseline ({NUM_EVAL_EPISODES} episodes)...")
random_results = evaluate_random(env, num_episodes=NUM_EVAL_EPISODES)

# -------------------------
# Results Summary
# -------------------------
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print(f"\n{'Method':<20} {'Mean Distance':<15} {'Std Dev':<12} {'Min':<10} {'Max':<10}")
print("-" * 70)
print(f"{'DQN':<20} {dqn_results['mean_distance']:<15.2f} {dqn_results['std_distance']:<12.2f} "
      f"{dqn_results['min_distance']:<10.2f} {dqn_results['max_distance']:<10.2f}")
print(f"{'Greedy (Baseline)':<20} {greedy_results['mean_distance']:<15.2f} {greedy_results['std_distance']:<12.2f} "
      f"{greedy_results['min_distance']:<10.2f} {greedy_results['max_distance']:<10.2f}")
print(f"{'Random':<20} {random_results['mean_distance']:<15.2f} {random_results['std_distance']:<12.2f} "
      f"{random_results['min_distance']:<10.2f} {random_results['max_distance']:<10.2f}")

# -------------------------
# Improvement Metrics
# -------------------------
print("\n" + "=" * 70)
print("IMPROVEMENT ANALYSIS")
print("=" * 70)

improvement_vs_greedy = ((greedy_results['mean_distance'] - dqn_results['mean_distance']) / 
                         greedy_results['mean_distance'] * 100)
improvement_vs_random = ((random_results['mean_distance'] - dqn_results['mean_distance']) / 
                         random_results['mean_distance'] * 100)

print(f"\nDQN vs Greedy:  {improvement_vs_greedy:+.2f}% {'✅' if improvement_vs_greedy > 0 else '❌'}")
print(f"DQN vs Random:  {improvement_vs_random:+.2f}% {'✅' if improvement_vs_greedy > 0 else '❌'}")

if improvement_vs_greedy > 0:
    print(f"\n🎉 DQN beats greedy baseline by {improvement_vs_greedy:.2f}%!")
elif improvement_vs_greedy > -5:
    print(f"\n⚠️  DQN is competitive with greedy (within 5%)")
else:
    print(f"\n⚠️  DQN underperforms greedy baseline by {abs(improvement_vs_greedy):.2f}%")
    print("Consider training longer or adjusting hyperparameters")

# -------------------------
# Best Route Visualization
# -------------------------
# -------------------------
# Get Best Route Info (for saving/visualization)
# -------------------------
best_idx = np.argmin(dqn_results['distances'])
best_route = dqn_results['routes'][best_idx]
best_distance = dqn_results['distances'][best_idx]

print(f"\nBest DQN route found: {best_distance:.2f} (across {NUM_EVAL_EPISODES} episodes)")
print(f"Route has {len(best_route)-1} stops")

# -------------------------
# Save Results
# -------------------------
results_summary = {
    'dqn': dqn_results,
    'greedy': greedy_results,
    'random': random_results,
    'improvement_vs_greedy': improvement_vs_greedy,
    'improvement_vs_random': improvement_vs_random,
    'best_route': best_route,
    'best_distance': best_distance
}

import pickle
results_path = "../resultimage/evaluation_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results_summary, f)
print(f"\n💾 Results saved to {results_path}")

# -------------------------
# Optional: Visualize Best Route
# -------------------------
try:
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract coordinates from best route
    route_x = [loc[1] for loc in best_route]
    route_y = [loc[2] for loc in best_route]
    
    # Plot all bins
    all_bins_coords = list(env.bin_lookup.values())
    all_x = [c[0] for c in all_bins_coords]
    all_y = [c[1] for c in all_bins_coords]
    ax.scatter(all_x, all_y, c='lightgray', s=100, label='All Bins', zorder=1)
    
    # Plot route path
    ax.plot(route_x, route_y, 'b-', linewidth=2, alpha=0.6, label='DQN Route', zorder=2)
    ax.scatter(route_x[1:], route_y[1:], c='red', s=100, label='Visited Bins', zorder=3)
    
    # Mark start and end
    ax.scatter([0], [0], c='green', s=300, marker='s', label='Start (Dispatch)', zorder=4)
    ax.scatter([route_x[-1]], [route_y[-1]], c='purple', s=300, marker='*', 
              label='End', zorder=4)
    
    # Add step numbers
    for i, (bin_id, x, y) in enumerate(best_route[1:], 1):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
    
    ax.set_xlim(-0.5, env.grid_size + 0.5)
    ax.set_ylim(-0.5, env.grid_size + 0.5)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Best DQN Route\nTotal Distance: {best_distance:.2f}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    viz_path = '../resultimage/best_route_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"📊 Best route visualization saved to {viz_path}")
    
except ImportError:
    print("⚠️  Matplotlib not available - skipping visualization")

print("\n" + "=" * 70)
print("✅ Evaluation complete!")
print("=" * 70)