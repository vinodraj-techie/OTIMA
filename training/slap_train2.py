# slap_train_fixed.py
import torch
import pandas as pd
import numpy as np
import os
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from model_architecture.slap import ImprovedSLAPGNN

DATA_PATH = "../synthetic_data_scripts/derived_data/"
MODEL_PATH = "../models/slap_fixed/"
os.makedirs(MODEL_PATH, exist_ok=True)

# -------------------------
# Hyperparameters
# -------------------------
HIDDEN_DIM = 128
NUM_LAYERS = 3
LEARNING_RATE = 0.003
EPOCHS = 300
PATIENCE = 30
BATCH_SIZE = 256
MARGIN = 1.0

# Loss weights
TRIPLET_WEIGHT = 0.3      # Contrastive learning
PHYSICAL_WEIGHT = 1.0     # Physical distance (primary objective)
BALANCE_WEIGHT = 0.5      # Bin utilization balance
VELOCITY_WEIGHT = 0.3     # High-velocity SKUs → near bins

TEMPERATURE = 0.1  # For soft assignment
MAX_CAPACITY = 3   # Soft capacity per bin (avg SKUs per bin target)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# -------------------------
# Load Data
# -------------------------
print("\nLoading data...")
sku_df = pd.read_csv(DATA_PATH + "sku_node_features_scaled.csv")
bin_df_scaled = pd.read_csv(DATA_PATH + "bin_node_features_scaled.csv")
edges_df = pd.read_csv(DATA_PATH + "sku_bin_edges.csv")

# Load UNSCALED bin data for physical distances
bin_df_unscaled = pd.read_csv(DATA_PATH + "bin_node_features.csv")

# Ensure physical distances exist
if "distance_from_dispatch" not in bin_df_unscaled.columns:
    print("Computing distances from coordinates...")
    bin_df_unscaled["distance_from_dispatch"] = np.sqrt(
        bin_df_unscaled["x_coord"]**2 + bin_df_unscaled["y_coord"]**2
    )

print(f"SKUs: {len(sku_df)}, Bins: {len(bin_df_scaled)}, Edges: {len(edges_df)}")
print(f"Physical distance range: {bin_df_unscaled['distance_from_dispatch'].min():.2f} to "
      f"{bin_df_unscaled['distance_from_dispatch'].max():.2f}")

# -------------------------
# Extract SKU velocities for velocity-aware assignment
# -------------------------
sku_numeric_df = sku_df.drop(columns=["sku_id"]).select_dtypes(include=[np.number])
if "total_quantity" in sku_numeric_df.columns:
    sku_velocities = sku_numeric_df["total_quantity"].values
    has_velocity = True
    print(f"✅ Using SKU velocities (range: {sku_velocities.min():.2f} to {sku_velocities.max():.2f})")
else:
    sku_velocities = np.ones(len(sku_df))
    has_velocity = False
    print(f"⚠️  No velocity data found, treating all SKUs equally")

# Normalize velocities
sku_velocities = (sku_velocities - sku_velocities.min()) / (sku_velocities.max() - sku_velocities.min() + 1e-8)
sku_velocities = torch.tensor(sku_velocities, dtype=torch.float).to(DEVICE)

# -------------------------
# Physical distances tensor
# -------------------------
physical_distances = torch.tensor(
    bin_df_unscaled["distance_from_dispatch"].values, 
    dtype=torch.float
).to(DEVICE)

# Normalize for loss computation
physical_distances_norm = physical_distances / physical_distances.max()

# -------------------------
# Node Features
# -------------------------
bin_numeric_df = bin_df_scaled.drop(columns=["bin_id"]).select_dtypes(include=[np.number])

sku_x = torch.tensor(sku_numeric_df.values, dtype=torch.float).to(DEVICE)
bin_x = torch.tensor(bin_numeric_df.values, dtype=torch.float).to(DEVICE)

num_skus = sku_x.shape[0]
num_bins = bin_x.shape[0]

# -------------------------
# Encode IDs
# -------------------------
sku_map = dict(zip(sku_df["sku_id"], range(num_skus)))
bin_map = dict(zip(bin_df_scaled["bin_id"], range(num_bins)))

edges_df["sku_idx"] = edges_df["sku_id"].map(sku_map)
edges_df["bin_idx"] = edges_df["bin_id"].map(bin_map) + num_skus

# -------------------------
# Train / Val Split
# -------------------------
train_edges, val_edges = train_test_split(edges_df, test_size=0.2, random_state=42)

def build_edge_index(df):
    edges = []
    for _, r in df.iterrows():
        edges.append([r["sku_idx"], r["bin_idx"]])
        edges.append([r["bin_idx"], r["sku_idx"]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

train_edge_index = build_edge_index(train_edges).to(DEVICE)
val_edge_index = build_edge_index(val_edges).to(DEVICE)

# -------------------------
# Precompute near/far bins for triplet loss
# -------------------------
bin_distances_np = bin_df_unscaled["distance_from_dispatch"].values
sorted_bins = np.argsort(bin_distances_np)
near_bins = sorted_bins[:int(0.2 * len(bin_distances_np))]
far_bins = sorted_bins[-int(0.2 * len(bin_distances_np)):]

# -------------------------
# Improved GNN Model
# -------------------------


# -------------------------
# Initialize Model
# -------------------------
model = ImprovedSLAPGNN(
    sku_dim=sku_x.shape[1],
    bin_dim=bin_x.shape[1],
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(DEVICE)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------
# Multi-Objective Loss Function
# -------------------------
def compute_combined_loss(embeddings, num_skus, num_bins, physical_distances, 
                          sku_velocities, near_bins, far_bins, 
                          temperature=0.1, num_samples=256):
    """
    Combined loss that optimizes:
    1. Physical distance (primary)
    2. Triplet/contrastive learning
    3. Bin balance
    4. Velocity-aware placement
    """
    
    sku_embeddings = embeddings[:num_skus]
    bin_embeddings = embeddings[num_skus:]
    
    # === 1. PHYSICAL DISTANCE LOSS (Primary Objective) ===
    # Compute soft assignments using Gumbel-Softmax
    emb_distances = torch.cdist(sku_embeddings, bin_embeddings, p=2)
    
    # Soft assignment probabilities (differentiable)
    assignment_probs = torch.softmax(-emb_distances / temperature, dim=1)
    
    # Expected physical distance for each SKU (weighted by assignment probability)
    expected_physical_dists = torch.matmul(assignment_probs, physical_distances)
    
    # Velocity-weighted: high-velocity SKUs penalized more for far distances
    if has_velocity:
        velocity_weights = 1.0 + 2.0 * sku_velocities  # High velocity → weight up to 3x
        weighted_physical_dists = expected_physical_dists * velocity_weights
    else:
        weighted_physical_dists = expected_physical_dists
    
    physical_loss = weighted_physical_dists.mean()
    
    # === 2. TRIPLET LOSS (Contrastive Learning) ===
    # Sample-based for efficiency
    sku_indices = np.random.choice(num_skus, size=min(num_samples, num_skus), replace=False)
    positive_bins = np.random.choice(near_bins, size=len(sku_indices))
    negative_bins = np.random.choice(far_bins, size=len(sku_indices))
    
    anchor_emb = embeddings[sku_indices]
    positive_emb = embeddings[num_skus + positive_bins]
    negative_emb = embeddings[num_skus + negative_bins]
    
    positive_dist = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
    negative_dist = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)
    
    triplet_loss = torch.relu(positive_dist - negative_dist + MARGIN).mean()
    
    # === 3. BALANCE/DIVERSITY LOSS ===
    # Encourage uniform distribution across bins
    bin_assignment_counts = assignment_probs.sum(dim=0)  # Sum over all SKUs
    bin_probs = bin_assignment_counts / bin_assignment_counts.sum()
    
    # Entropy (higher = more balanced)
    entropy = -(bin_probs * torch.log(bin_probs + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(num_bins, dtype=torch.float))
    
    # Loss = deviation from max entropy (0 = perfectly balanced)
    balance_loss = (max_entropy - entropy) / max_entropy
    
    # === 4. CAPACITY CONSTRAINT (Soft) ===
    # Penalize bins that exceed target capacity
    target_per_bin = num_skus / num_bins
    capacity_violations = torch.relu(bin_assignment_counts - MAX_CAPACITY * target_per_bin)
    capacity_loss = capacity_violations.sum() / num_bins
    
    # === COMBINED LOSS ===
    total_loss = (
        PHYSICAL_WEIGHT * physical_loss +
        TRIPLET_WEIGHT * triplet_loss +
        BALANCE_WEIGHT * balance_loss +
        BALANCE_WEIGHT * capacity_loss * 0.5  # Half weight for capacity
    )
    
    return total_loss, {
        'physical': physical_loss.item(),
        'triplet': triplet_loss.item(),
        'balance': balance_loss.item(),
        'capacity': capacity_loss.item(),
        'total': total_loss.item()
    }

# -------------------------
# Validation Function
# -------------------------
def validate(model, sku_x, bin_x, edge_index):
    model.eval()
    with torch.no_grad():
        emb = model(sku_x, bin_x, edge_index)
        val_loss, loss_components = compute_combined_loss(
            emb, num_skus, num_bins, physical_distances_norm, 
            sku_velocities, near_bins, far_bins
        )
    return val_loss.item(), loss_components

# -------------------------
# Training Loop
# -------------------------
print("\nStarting training with multi-objective loss...")
print("="*70)
print(f"Loss weights: Physical={PHYSICAL_WEIGHT}, Triplet={TRIPLET_WEIGHT}, "
      f"Balance={BALANCE_WEIGHT}, Velocity={'Yes' if has_velocity else 'No'}")
print("="*70)

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []
loss_history = {
    'physical': [], 'triplet': [], 'balance': [], 'capacity': []
}

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_components = {'physical': 0, 'triplet': 0, 'balance': 0, 'capacity': 0}
    num_batches = max(1, num_skus // BATCH_SIZE)
    
    for batch in range(num_batches):
        optimizer.zero_grad()
        
        emb = model(sku_x, bin_x, train_edge_index)
        
        loss, components = compute_combined_loss(
            emb, num_skus, num_bins, physical_distances_norm,
            sku_velocities, near_bins, far_bins,
            temperature=TEMPERATURE, num_samples=BATCH_SIZE
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        for key in epoch_components:
            epoch_components[key] += components[key]
    
    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Store component losses
    for key in epoch_components:
        loss_history[key].append(epoch_components[key] / num_batches)
    
    # Validation
    val_loss, val_components = validate(model, sku_x, bin_x, val_edge_index)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    # Logging
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Phys: {val_components['physical']:.4f} | Bal: {val_components['balance']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, MODEL_PATH + "slap_gnn_best.pt")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print("\n" + "="*70)
print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

# Load best model
checkpoint = torch.load(MODEL_PATH + "slap_gnn_best.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# -------------------------
# Inference with Hard Assignment
# -------------------------
print("\nPerforming inference...")
model.eval()

with torch.no_grad():
    emb = model(sku_x, bin_x, train_edge_index)
    
    sku_embeddings = emb[:num_skus]
    bin_embeddings = emb[num_skus:]
    
    # Hard assignment: nearest bin in embedding space
    distances = torch.cdist(sku_embeddings, bin_embeddings, p=2)
    best_bin_indices = torch.argmin(distances, dim=1).cpu().numpy()

# Create assignments
assignments = [bin_df_scaled.iloc[idx]["bin_id"] for idx in best_bin_indices]

# Get actual physical distances
physical_dists_assigned = []
for idx in best_bin_indices:
    phys_dist = bin_df_unscaled.iloc[idx]["distance_from_dispatch"]
    physical_dists_assigned.append(phys_dist)

out = pd.DataFrame({
    "sku_id": sku_df["sku_id"],
    "assigned_bin": assignments,
    "embedding_distance": distances[range(num_skus), best_bin_indices].cpu().numpy(),
    "physical_distance": physical_dists_assigned
})

out.to_csv(DATA_PATH + "optimized_storage.csv", index=False)

# -------------------------
# Save Training History
# -------------------------
history = pd.DataFrame({
    'epoch': range(len(train_losses)),
    'train_loss': train_losses,
    'val_loss': val_losses,
    'physical_loss': loss_history['physical'],
    'triplet_loss': loss_history['triplet'],
    'balance_loss': loss_history['balance'],
    'capacity_loss': loss_history['capacity']
})
history.to_csv(MODEL_PATH + "training_history.csv", index=False)

# Save metadata
import json
metadata = {
    'num_skus': num_skus,
    'num_bins': num_bins,
    'hidden_dim': HIDDEN_DIM,
    'num_layers': NUM_LAYERS,
    'best_epoch': int(checkpoint['epoch']),
    'best_val_loss': float(best_val_loss),
    'total_epochs': len(train_losses),
    'loss_weights': {
        'physical': PHYSICAL_WEIGHT,
        'triplet': TRIPLET_WEIGHT,
        'balance': BALANCE_WEIGHT,
        'velocity': VELOCITY_WEIGHT
    },
    'has_velocity_data': has_velocity
}

with open(MODEL_PATH + "model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# -------------------------
# Quick Statistics
# -------------------------
print("\n" + "="*70)
print("QUICK STATISTICS")
print("="*70)

avg_physical_dist = out["physical_distance"].mean()
avg_emb_dist = out["embedding_distance"].mean()
bin_usage = out["assigned_bin"].value_counts()

print(f"Average PHYSICAL distance: {avg_physical_dist:.2f}")
print(f"Average embedding distance: {avg_emb_dist:.4f}")
print(f"Bins used: {len(bin_usage)} / {num_bins} ({len(bin_usage)/num_bins*100:.1f}%)")
print(f"Avg SKUs per bin: {num_skus / len(bin_usage):.2f}")
print(f"Most loaded bin: {bin_usage.max()} SKUs")
print(f"Least loaded bin: {bin_usage.min()} SKUs")
print(f"Imbalance ratio: {bin_usage.max() / bin_usage.min():.2f}")

# Velocity analysis
if has_velocity:
    high_velocity_skus = sku_velocities.cpu().numpy() > 0.7
    high_vel_dist = out.loc[high_velocity_skus, "physical_distance"].mean()
    low_vel_dist = out.loc[~high_velocity_skus, "physical_distance"].mean()
    print(f"\nVelocity-aware placement:")
    print(f"  High-velocity SKUs avg dist: {high_vel_dist:.2f}")
    print(f"  Low-velocity SKUs avg dist:  {low_vel_dist:.2f}")
    print(f"  {'✅ High-velocity closer' if high_vel_dist < low_vel_dist else '❌ No correlation'}")

print("\n✅ SLAP training completed successfully")
print(f"   Best model: {MODEL_PATH}slap_gnn_best.pt")
print(f"   Assignments: {DATA_PATH}optimized_storage.csv")
print(f"   History: {MODEL_PATH}training_history.csv")