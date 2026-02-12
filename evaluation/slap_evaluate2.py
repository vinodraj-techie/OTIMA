# slap_evaluation_final.py
import torch
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.nn import GATConv
from sklearn.metrics import silhouette_score
from collections import Counter

DATA_PATH = "../synthetic_data_scripts/derived_data/"
MODEL_PATH = "../models/slap_fixed/"  # ← Changed path
RESULTS_PATH = "../resultimage/slap/"

os.makedirs(RESULTS_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("FIXED SLAP GNN - COMPREHENSIVE EVALUATION")
print("="*70)

# -------------------------
# Load Metadata
# -------------------------
with open(MODEL_PATH + "model_metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"\nModel Information:")
print(f"  SKUs: {metadata['num_skus']}")
print(f"  Bins: {metadata['num_bins']}")
print(f"  Best Epoch: {metadata['best_epoch']}")
print(f"  Best Val Loss: {metadata['best_val_loss']:.4f}")
print(f"  Has Velocity Data: {metadata.get('has_velocity_data', 'Unknown')}")

if 'loss_weights' in metadata:
    print(f"\n  Loss Weights:")
    for key, val in metadata['loss_weights'].items():
        print(f"    {key}: {val}")

# -------------------------
# Load Data
# -------------------------
sku_df = pd.read_csv(DATA_PATH + "sku_node_features_scaled.csv")
bin_df_unscaled = pd.read_csv(DATA_PATH + "bin_node_features.csv")
assignments_df = pd.read_csv(DATA_PATH + "optimized_storage.csv")
training_history = pd.read_csv(MODEL_PATH + "training_history.csv")

# Ensure physical distances exist
if "distance_from_dispatch" not in bin_df_unscaled.columns:
    bin_df_unscaled["distance_from_dispatch"] = np.sqrt(
        bin_df_unscaled["x_coord"]**2 + bin_df_unscaled["y_coord"]**2
    )

print(f"\nData loaded:")
print(f"  Assignments: {len(assignments_df)}")
print(f"  Training epochs: {len(training_history)}")

# Check if new columns exist
has_physical_col = "physical_distance" in assignments_df.columns
has_component_losses = "physical_loss" in training_history.columns

print(f"  Physical distance in output: {'✅ Yes' if has_physical_col else '❌ No (old format)'}")
print(f"  Loss components tracked: {'✅ Yes' if has_component_losses else '❌ No (old format)'}")

# -------------------------
# Get Physical Distances
# -------------------------
if has_physical_col:
    # New format: already in CSV
    physical_distances = assignments_df["physical_distance"].values
else:
    # Old format: merge to get them
    assignments_df = assignments_df.merge(
        bin_df_unscaled[["bin_id", "distance_from_dispatch"]], 
        left_on="assigned_bin",
        right_on="bin_id",
        how="left"
    )
    physical_distances = assignments_df["distance_from_dispatch"].values

# -------------------------
# Distance Metrics
# -------------------------
print("\n" + "="*70)
print("DISTANCE METRICS")
print("="*70)

avg_physical = np.mean(physical_distances)
median_physical = np.median(physical_distances)
std_physical = np.std(physical_distances)

print(f"Physical Distance (what matters):")
print(f"  Average:  {avg_physical:.2f} meters")
print(f"  Median:   {median_physical:.2f} meters")
print(f"  Std Dev:  {std_physical:.2f} meters")

if "embedding_distance" in assignments_df.columns:
    avg_embedding = assignments_df["embedding_distance"].mean()
    print(f"\nEmbedding Distance (learned space):")
    print(f"  Average:  {avg_embedding:.4f}")
    
    # Correlation
    corr = np.corrcoef(physical_distances, assignments_df["embedding_distance"])[0, 1]
    print(f"\nCorrelation (Emb ↔ Physical): {corr:.3f}")
    if corr > 0.5:
        print(f"  ✅ Strong positive correlation - embeddings respect physical layout")
    elif corr > 0.3:
        print(f"  ⚠️  Moderate correlation")
    else:
        print(f"  ❌ Weak correlation - embeddings don't match physical space")

# -------------------------
# Baseline Comparisons
# -------------------------
def random_assignment():
    np.random.seed(42)
    assignments = np.random.choice(bin_df_unscaled["bin_id"], size=len(sku_df))
    random_df = pd.DataFrame({"assigned_bin": assignments})
    random_df = random_df.merge(
        bin_df_unscaled[["bin_id", "distance_from_dispatch"]], 
        left_on="assigned_bin", right_on="bin_id"
    )
    return random_df["distance_from_dispatch"].mean()

def nearest_bin_assignment():
    sorted_bins = bin_df_unscaled.sort_values("distance_from_dispatch")
    nearest_bins = sorted_bins.head(len(sku_df))["bin_id"].values
    nearest_df = pd.DataFrame({"assigned_bin": nearest_bins})
    nearest_df = nearest_df.merge(
        bin_df_unscaled[["bin_id", "distance_from_dispatch"]], 
        left_on="assigned_bin", right_on="bin_id"
    )
    return nearest_df["distance_from_dispatch"].mean()

def velocity_based_assignment():
    sku_numeric = sku_df.drop(columns=["sku_id"]).select_dtypes(include=[np.number])
    
    if "total_quantity" in sku_numeric.columns:
        velocities = sku_numeric["total_quantity"].values
    else:
        velocities = np.ones(len(sku_df))
    
    sku_order = np.argsort(velocities)[::-1]
    bin_order = bin_df_unscaled.sort_values("distance_from_dispatch")["bin_id"].values
    
    assignments = [None] * len(sku_df)
    for i, sku_idx in enumerate(sku_order):
        assignments[sku_idx] = bin_order[i % len(bin_order)]
    
    velocity_df = pd.DataFrame({"assigned_bin": assignments})
    velocity_df = velocity_df.merge(
        bin_df_unscaled[["bin_id", "distance_from_dispatch"]], 
        left_on="assigned_bin", right_on="bin_id"
    )
    return velocity_df["distance_from_dispatch"].mean()

print("\n" + "="*70)
print("BASELINE COMPARISONS")
print("="*70)

random_avg = random_assignment()
nearest_avg = nearest_bin_assignment()
velocity_avg = velocity_based_assignment()

print(f"Random Assignment:       {random_avg:.2f}")
print(f"Nearest Bin (Greedy):    {nearest_avg:.2f}")
print(f"Velocity-Based:          {velocity_avg:.2f}")
print(f"Fixed SLAP GNN:          {avg_physical:.2f}")

improvement_vs_random = ((random_avg - avg_physical) / random_avg) * 100
improvement_vs_nearest = ((nearest_avg - avg_physical) / nearest_avg) * 100
improvement_vs_velocity = ((velocity_avg - avg_physical) / velocity_avg) * 100

print("\n" + "="*70)
print("IMPROVEMENT ANALYSIS")
print("="*70)
print(f"vs Random:        {improvement_vs_random:+.2f}% {'✅' if improvement_vs_random > 0 else '❌'}")
print(f"vs Nearest:       {improvement_vs_nearest:+.2f}% {'✅' if improvement_vs_nearest > 0 else '❌'}")
print(f"vs Velocity:      {improvement_vs_velocity:+.2f}% {'✅' if improvement_vs_velocity > 0 else '❌'}")

# -------------------------
# Bin Utilization
# -------------------------
print("\n" + "="*70)
print("BIN UTILIZATION & BALANCE")
print("="*70)

bin_usage = Counter(assignments_df["assigned_bin"])
used_bins = len(bin_usage)
unused_bins = len(bin_df_unscaled) - used_bins
utilization_rate = (used_bins / len(bin_df_unscaled)) * 100

print(f"Bins used:         {used_bins} / {len(bin_df_unscaled)} ({utilization_rate:.1f}%)")
print(f"Bins unused:       {unused_bins}")

skus_per_bin = list(bin_usage.values())
avg_skus = np.mean(skus_per_bin)
max_skus = np.max(skus_per_bin)
min_skus = np.min(skus_per_bin)
std_skus = np.std(skus_per_bin)

print(f"Avg SKUs per bin:  {avg_skus:.2f}")
print(f"Max SKUs in bin:   {max_skus}")
print(f"Min SKUs in bin:   {min_skus}")
print(f"Std dev:           {std_skus:.2f}")

imbalance_ratio = max_skus / min_skus if min_skus > 0 else float('inf')
print(f"Imbalance ratio:   {imbalance_ratio:.2f} {'✅ Good' if imbalance_ratio < 3 else '⚠️ High' if imbalance_ratio < 5 else '❌ Very High'}")

# Gini coefficient
sorted_usage = np.sort(skus_per_bin)
n = len(sorted_usage)
cumsum = np.cumsum(sorted_usage)
gini = (2 * np.sum((np.arange(1, n+1)) * sorted_usage)) / (n * cumsum[-1]) - (n + 1) / n
print(f"Gini coefficient:  {gini:.4f} (0=perfect, 1=max imbalance)")

# -------------------------
# Velocity Analysis
# -------------------------
sku_numeric = sku_df.drop(columns=["sku_id"]).select_dtypes(include=[np.number])
if "total_quantity" in sku_numeric.columns:
    print("\n" + "="*70)
    print("VELOCITY-AWARE PLACEMENT")
    print("="*70)
    
    velocities = sku_numeric["total_quantity"].values
    high_velocity_mask = velocities > np.percentile(velocities, 70)
    
    high_vel_dist = physical_distances[high_velocity_mask].mean()
    low_vel_dist = physical_distances[~high_velocity_mask].mean()
    
    print(f"High-velocity SKUs (top 30%):")
    print(f"  Count: {high_velocity_mask.sum()}")
    print(f"  Avg distance: {high_vel_dist:.2f}")
    
    print(f"\nLow-velocity SKUs (bottom 70%):")
    print(f"  Count: {(~high_velocity_mask).sum()}")
    print(f"  Avg distance: {low_vel_dist:.2f}")
    
    if high_vel_dist < low_vel_dist:
        improvement = ((low_vel_dist - high_vel_dist) / low_vel_dist) * 100
        print(f"\n✅ High-velocity SKUs are {improvement:.1f}% closer (GOOD!)")
    else:
        print(f"\n❌ No velocity-distance correlation")

# -------------------------
# Visualization 1: Training Progress
# -------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall loss
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(training_history['epoch'], training_history['train_loss'], label='Train', linewidth=2)
ax1.plot(training_history['epoch'], training_history['val_loss'], label='Val', linewidth=2)
ax1.axvline(metadata['best_epoch'], color='r', linestyle='--', alpha=0.5, label='Best Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss')
ax1.set_title('Training Progress', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Component losses (if available)
if has_component_losses:
    ax2 = fig.add_subplot(gs[0, 2])
    final_components = {
        'Physical': training_history['physical_loss'].iloc[-1],
        'Triplet': training_history['triplet_loss'].iloc[-1],
        'Balance': training_history['balance_loss'].iloc[-1],
        'Capacity': training_history['capacity_loss'].iloc[-1]
    }
    ax2.bar(final_components.keys(), final_components.values(), color=['red', 'orange', 'green', 'blue'])
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Final Loss Components', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

# Method comparison
ax3 = fig.add_subplot(gs[1, 0])
methods = ['Random', 'Nearest', 'Velocity', 'SLAP GNN']
distances = [random_avg, nearest_avg, velocity_avg, avg_physical]
colors = ['gray', 'orange', 'yellow', 'green' if avg_physical == min(distances) else 'red']
bars = ax3.bar(methods, distances, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Avg Distance')
ax3.set_title('Method Comparison', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, dist in zip(bars, distances):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height*1.02,
            f'{dist:.1f}', ha='center', fontsize=9, fontweight='bold')

# Distance distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(physical_distances, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(avg_physical, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_physical:.2f}')
ax4.axvline(bin_df_unscaled["distance_from_dispatch"].mean(), color='g', 
           linestyle='--', linewidth=2, label='All Bins')
ax4.set_xlabel('Physical Distance')
ax4.set_ylabel('Count')
ax4.set_title('Assigned Distance Distribution', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Bin utilization
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(skus_per_bin, bins=min(20, max_skus), color='coral', edgecolor='black', alpha=0.7)
ax5.axvline(avg_skus, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_skus:.1f}')
ax5.set_xlabel('SKUs per Bin')
ax5.set_ylabel('Count')
ax5.set_title(f'Balance (Gini: {gini:.3f})', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Spatial heatmap
ax6 = fig.add_subplot(gs[2, :2])
bin_coords = bin_df_unscaled[["x_coord", "y_coord"]].values
bin_counts = []
for _, row in bin_df_unscaled.iterrows():
    count = len(assignments_df[assignments_df["assigned_bin"] == row["bin_id"]])
    bin_counts.append(count)

scatter = ax6.scatter(bin_coords[:, 0], bin_coords[:, 1], 
                     c=bin_counts, cmap='YlOrRd', s=50, alpha=0.6, 
                     edgecolors='black', linewidth=0.5, vmin=0)
ax6.scatter([0], [0], c='blue', s=300, marker='*', label='Dispatch', zorder=10)
plt.colorbar(scatter, ax=ax6, label='SKUs Assigned')
ax6.set_xlabel('X Coordinate')
ax6.set_ylabel('Y Coordinate')
ax6.set_title('Assignment Heatmap', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Improvement summary
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
summary_text = f"""
SUMMARY

Physical Distance: {avg_physical:.2f}m

vs Random: {improvement_vs_random:+.1f}%
vs Nearest: {improvement_vs_nearest:+.1f}%
vs Velocity: {improvement_vs_velocity:+.1f}%

Utilization: {utilization_rate:.1f}%
Imbalance: {imbalance_ratio:.2f}x
Gini: {gini:.3f}

Converged: Epoch {metadata['best_epoch']}
"""
ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('FIXED SLAP GNN - Comprehensive Evaluation', fontsize=16, fontweight='bold')
plt.savefig(RESULTS_PATH + 'slap_final_evaluation.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Comprehensive evaluation saved: {RESULTS_PATH}slap_final_evaluation.png")

# -------------------------
# Save Results
# -------------------------
results_summary = {
    'model_info': metadata,
    'distance_metrics': {
        'physical_avg': float(avg_physical),
        'physical_median': float(median_physical),
        'physical_std': float(std_physical),
    },
    'baselines': {
        'random': float(random_avg),
        'nearest': float(nearest_avg),
        'velocity': float(velocity_avg),
        'slap_gnn': float(avg_physical)
    },
    'improvements': {
        'vs_random': float(improvement_vs_random),
        'vs_nearest': float(improvement_vs_nearest),
        'vs_velocity': float(improvement_vs_velocity)
    },
    'utilization': {
        'bins_used': int(used_bins),
        'utilization_rate': float(utilization_rate),
        'avg_skus_per_bin': float(avg_skus),
        'imbalance_ratio': float(imbalance_ratio),
        'gini_coefficient': float(gini)
    }
}

with open(RESULTS_PATH + 'slap_final_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"💾 Results summary saved: {RESULTS_PATH}slap_final_results.json")

# -------------------------
# Final Assessment
# -------------------------
print("\n" + "="*70)
print("FINAL ASSESSMENT")
print("="*70)

score = 0
max_score = 5

# Check 1: Better than random
if improvement_vs_random > 20:
    print("✅ Much better than random (+20%)")
    score += 1
elif improvement_vs_random > 0:
    print("⚠️  Beats random but marginally")
    score += 0.5
else:
    print("❌ Worse than random")

# Check 2: Competitive with nearest
if improvement_vs_nearest > 10:
    print("✅ Significantly better than nearest-bin (+10%)")
    score += 1
elif improvement_vs_nearest > 0:
    print("✅ Beats nearest-bin baseline")
    score += 0.5
else:
    print("⚠️  Not beating simple nearest-bin")

# Check 3: Balance
if imbalance_ratio < 2:
    print("✅ Excellent balance (ratio < 2)")
    score += 1
elif imbalance_ratio < 3:
    print("✅ Good balance (ratio < 3)")
    score += 0.7
elif imbalance_ratio < 5:
    print("⚠️  Moderate imbalance (ratio < 5)")
    score += 0.3
else:
    print("❌ High imbalance (ratio ≥ 5)")

# Check 4: Velocity awareness
if "total_quantity" in sku_numeric.columns and high_vel_dist < low_vel_dist:
    vel_improvement = ((low_vel_dist - high_vel_dist) / low_vel_dist) * 100
    if vel_improvement > 10:
        print(f"✅ Strong velocity-distance correlation (-{vel_improvement:.1f}%)")
        score += 1
    else:
        print(f"✅ Velocity-aware placement (-{vel_improvement:.1f}%)")
        score += 0.5
else:
    print("⚠️  No velocity-distance correlation (or no data)")

# Check 5: Training stability
if metadata['best_epoch'] < metadata['total_epochs'] - 10:
    print("✅ Model converged with early stopping")
    score += 1
elif metadata['best_epoch'] == metadata['total_epochs'] - 1:
    print("⚠️  May need more training epochs")
    score += 0.3
else:
    print("✅ Training completed successfully")
    score += 0.7

# Final grade
print(f"\n{'='*70}")
print(f"OVERALL SCORE: {score:.1f} / {max_score}")

if score >= 4.5:
    grade = "🏆 EXCELLENT - Production Ready!"
elif score >= 3.5:
    grade = "✅ GOOD - Minor improvements possible"
elif score >= 2.5:
    grade = "⚠️  FAIR - Needs optimization"
else:
    grade = "❌ POOR - Requires major fixes"

print(f"GRADE: {grade}")
print("="*70)

print("\n✅ Evaluation complete!")