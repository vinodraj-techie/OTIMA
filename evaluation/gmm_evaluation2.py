import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

MODEL_PATH = "../models/gmm2/"
OUTPUT_PATH = "../synthetic_data_scripts/derived_data/"
RESULTS_PATH = "../resultimage/gmm/"

os.makedirs(RESULTS_PATH, exist_ok=True)

print("="*70)
print("OPTIMIZED GMM CLUSTERING - EVALUATION")
print("="*70)

# -----------------------
# Load Model and Metadata
# -----------------------
if not os.path.exists(MODEL_PATH + "gmm_model.pkl"):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please train the model first using gmm_train_optimized.py")
    exit(1)

gmm_model = joblib.load(MODEL_PATH + "gmm_model.pkl")
scaler = joblib.load(MODEL_PATH + "gmm_scaler.pkl")

# Load metadata
with open(MODEL_PATH + "gmm_metadata.json", 'r') as f:
    metadata = json.load(f)

use_pca = metadata.get('use_pca', False)
if use_pca:
    pca_model = joblib.load(MODEL_PATH + "gmm_pca.pkl")
    print("✅ PCA transformation detected")

# Load cluster assignments
clusters_df = pd.read_csv(OUTPUT_PATH + "gmm_clusters.csv")

# Load original features
feature_df = pd.read_csv("../synthetic_data_scripts/derived_data/gmm_features.csv")

# Get the features used during training
if 'features_used' in metadata:
    # Use the exact features that were used during training
    features_used = metadata['features_used']
    print(f"✅ Loading {len(features_used)} features used during training")
    X_original = feature_df[features_used]
else:
    # Fallback: try to match scaler's feature names
    print("⚠️  No feature list in metadata, attempting to match scaler features...")
    try:
        # sklearn scalers store feature_names_in_ attribute
        if hasattr(scaler, 'feature_names_in_'):
            features_used = scaler.feature_names_in_.tolist()
            X_original = feature_df[features_used]
            print(f"✅ Matched {len(features_used)} features from scaler")
        else:
            print("❌ Cannot determine which features were used during training")
            print("Please retrain the model with the updated training script")
            exit(1)
    except Exception as e:
        print(f"❌ Error matching features: {e}")
        print("Please retrain the model with the updated training script")
        exit(1)

# Apply same preprocessing as training
X_scaled = scaler.transform(X_original)
if use_pca:
    X_scaled = pca_model.transform(X_scaled)

# Get model info
num_clusters = gmm_model.n_components
num_skus = len(clusters_df)
num_features_original = X_original.shape[1]
num_features_used = X_scaled.shape[1]

print(f"\nModel Information:")
print(f"  Clusters:            {num_clusters}")
print(f"  SKUs:                {num_skus:,}")
print(f"  Original features:   {num_features_original}")
print(f"  Features used:       {num_features_used}")
print(f"  Scaler type:         {metadata['scaler_type']}")
print(f"  Covariance type:     {metadata['best_covariance']}")
print(f"  Converged:           {'Yes' if metadata['converged'] else 'No'}")
print(f"  BIC Score:           {metadata['bic']:.2f}")
print(f"  AIC Score:           {metadata['aic']:.2f}")

# -----------------------
# Calculate Statistics
# -----------------------
probabilities = gmm_model.predict_proba(X_scaled)
max_probs = probabilities.max(axis=1)

cluster_stats = pd.DataFrame({
    'cluster': range(num_clusters),
    'size': [(clusters_df['gmm_cluster'] == i).sum() for i in range(num_clusters)],
    'percentage': [(clusters_df['gmm_cluster'] == i).sum() / num_skus * 100 for i in range(num_clusters)],
    'avg_confidence': [probabilities[clusters_df['gmm_cluster'] == i, i].mean() for i in range(num_clusters)],
    'min_confidence': [probabilities[clusters_df['gmm_cluster'] == i, i].min() for i in range(num_clusters)],
    'max_confidence': [probabilities[clusters_df['gmm_cluster'] == i, i].max() for i in range(num_clusters)]
})

print("\n" + "="*70)
print("CLUSTER STATISTICS")
print("="*70)
print(cluster_stats[['cluster', 'size', 'percentage', 'avg_confidence']].to_string(index=False))

# -----------------------
# Quality Metrics
# -----------------------
silhouette = silhouette_score(X_scaled, clusters_df['gmm_cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, clusters_df['gmm_cluster'])
calinski = calinski_harabasz_score(X_scaled, clusters_df['gmm_cluster'])

print("\n" + "="*70)
print("CLUSTERING QUALITY METRICS")
print("="*70)
print(f"Silhouette Score:        {silhouette:.4f}  (higher is better, range: -1 to 1)")
print(f"Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower is better)")
print(f"Calinski-Harabasz Score: {calinski:.0f}  (higher is better)")
print(f"Mean Confidence:         {max_probs.mean():.4f}")
print(f"Min Confidence:          {max_probs.min():.4f}")
print(f"Max Confidence:          {max_probs.max():.4f}")

# -----------------------
# Visualization 1: Cluster Distribution
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cluster size distribution
colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))
ax1.bar(cluster_stats['cluster'], cluster_stats['size'], 
        color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of SKUs', fontsize=12, fontweight='bold')
ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_xticks(range(num_clusters))

# Add value and percentage labels on bars
for i, row in cluster_stats.iterrows():
    height = row['size']
    ax1.text(row['cluster'], height + max(cluster_stats['size'])*0.01, 
            f"{int(height)}\n({row['percentage']:.1f}%)", 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Pie chart
explode = [0.05] * num_clusters  # slight separation
ax2.pie(cluster_stats['size'], 
        labels=[f"Cluster {i}\n({s} SKUs)" for i, s in enumerate(cluster_stats['size'])],
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        explode=explode,
        textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Cluster Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'gmm_cluster_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Cluster distribution saved: {RESULTS_PATH}gmm_cluster_distribution.png")
plt.close()

# -----------------------
# Visualization 2: Confidence Analysis
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Overall confidence distribution
ax1.hist(max_probs, bins=30, color='skyblue', edgecolor='black', alpha=0.7, linewidth=1.5)
ax1.axvline(max_probs.mean(), color='red', linestyle='--', linewidth=2.5, 
           label=f"Mean: {max_probs.mean():.3f}")
ax1.axvline(np.median(max_probs), color='green', linestyle='-.', linewidth=2.5, 
           label=f"Median: {np.median(max_probs):.3f}")
ax1.set_xlabel('Assignment Confidence', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Cluster Assignment Confidence Distribution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Confidence by cluster (boxplot)
cluster_confidences = [probabilities[clusters_df['gmm_cluster'] == i, i] 
                       for i in range(num_clusters)]
bp = ax2.boxplot(cluster_confidences, labels=range(num_clusters), patch_artist=True,
                 showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_linewidth(1.5)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], linewidth=1.5)
ax2.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax2.set_ylabel('Assignment Confidence', fontsize=12, fontweight='bold')
ax2.set_title('Confidence Distribution by Cluster', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'gmm_confidence_analysis.png', dpi=150, bbox_inches='tight')
print(f"📊 Confidence analysis saved: {RESULTS_PATH}gmm_confidence_analysis.png")
plt.close()

# -----------------------
# Visualization 3: PCA Projection (2D)
# -----------------------
# Always create a 2D PCA for visualization (even if model uses different dimensions)
pca_viz = PCA(n_components=2)
X_pca_viz = pca_viz.fit_transform(scaler.transform(X_original))

fig, ax = plt.subplots(figsize=(12, 9))
scatter = ax.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], 
                     c=clusters_df['gmm_cluster'], 
                     cmap='tab10', 
                     s=60, 
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)

# Add cluster centers (transform back to original space if PCA was used in training)
if use_pca:
    # Model means are in PCA space, need to inverse transform
    centers_pca_space = gmm_model.means_
    centers_original_space = pca_model.inverse_transform(centers_pca_space)
    centers_2d = pca_viz.transform(centers_original_space)
else:
    # Model means are already in scaled space
    centers_original_space = scaler.inverse_transform(gmm_model.means_)
    centers_2d = pca_viz.transform(scaler.transform(centers_original_space))

ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
          c='red', marker='X', s=300, edgecolors='black', linewidth=2,
          label='Cluster Centers', zorder=10)

# Add cluster labels near centers
for i, (x, y) in enumerate(centers_2d):
    ax.annotate(f'C{i}', (x, y), fontsize=12, fontweight='bold', 
               ha='center', va='center', color='white',
               bbox=dict(boxstyle='circle', facecolor='red', edgecolor='black', linewidth=2))

cbar = plt.colorbar(scatter, label='Cluster', ax=ax)
cbar.set_label('Cluster', fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}% variance)', 
             fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}% variance)', 
             fontsize=12, fontweight='bold')
ax.set_title('GMM Clusters - PCA 2D Projection', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'gmm_pca_projection.png', dpi=150, bbox_inches='tight')
print(f"📊 PCA projection saved: {RESULTS_PATH}gmm_pca_projection.png")
plt.close()

# -----------------------
# Visualization 4: Feature Profiles by Cluster
# -----------------------
feature_names = list(X_original.columns)

if len(feature_names) <= 15:
    # Show all features if reasonable number
    n_cols = min(num_clusters, 4)
    n_rows = int(np.ceil(num_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if num_clusters > 1 else [axes]
    
    for i in range(num_clusters):
        cluster_mask = clusters_df['gmm_cluster'] == i
        cluster_features = X_original[cluster_mask]
        
        feature_means = cluster_features.mean()
        feature_stds = cluster_features.std()
        
        y_pos = np.arange(len(feature_names))
        axes[i].barh(y_pos, feature_means, xerr=feature_stds, 
                    color=colors[i], edgecolor='black', alpha=0.7,
                    error_kw={'linewidth': 1.5, 'ecolor': 'darkgray'})
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(feature_names, fontsize=9)
        axes[i].set_xlabel('Mean Value ± Std', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Cluster {i}\n({cluster_stats.iloc[i]["size"]} SKUs)', 
                         fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Hide empty subplots
    for i in range(num_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'gmm_feature_profiles.png', dpi=150, bbox_inches='tight')
    print(f"📊 Feature profiles saved: {RESULTS_PATH}gmm_feature_profiles.png")
    plt.close()
elif len(feature_names) <= 30:
    # Show top features by variance
    feature_variance = X_original.var()
    top_features = feature_variance.nlargest(10).index.tolist()
    
    n_cols = min(num_clusters, 4)
    n_rows = int(np.ceil(num_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if num_clusters > 1 else [axes]
    
    for i in range(num_clusters):
        cluster_mask = clusters_df['gmm_cluster'] == i
        cluster_features = X_original.loc[cluster_mask, top_features]
        
        feature_means = cluster_features.mean()
        feature_stds = cluster_features.std()
        
        y_pos = np.arange(len(top_features))
        axes[i].barh(y_pos, feature_means, xerr=feature_stds, 
                    color=colors[i], edgecolor='black', alpha=0.7,
                    error_kw={'linewidth': 1.5, 'ecolor': 'darkgray'})
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(top_features, fontsize=8)
        axes[i].set_xlabel('Mean Value ± Std', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Cluster {i}\n({cluster_stats.iloc[i]["size"]} SKUs)\nTop 10 Features by Variance', 
                         fontsize=10, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Hide empty subplots
    for i in range(num_clusters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'gmm_feature_profiles_top10.png', dpi=150, bbox_inches='tight')
    print(f"📊 Top 10 feature profiles saved: {RESULTS_PATH}gmm_feature_profiles_top10.png")
    plt.close()
else:
    print(f"⚠️  Too many features ({len(feature_names)}) - skipping detailed feature profiles")

# -----------------------
# Visualization 5: Cluster Imbalance and Quality Dashboard
# -----------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Cluster sizes with imbalance indicator
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.bar(cluster_stats['cluster'], cluster_stats['size'], 
               color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of SKUs', fontsize=11, fontweight='bold')
ax1.set_title('Cluster Distribution & Imbalance', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.axhline(y=num_skus/num_clusters, color='red', linestyle='--', linewidth=2, 
           label=f'Balanced ({num_skus//num_clusters} per cluster)')
ax1.legend(fontsize=9)
for i, row in cluster_stats.iterrows():
    ax1.text(row['cluster'], row['size'], f"{int(row['size'])}", 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Quality metrics comparison
ax2 = fig.add_subplot(gs[0, 2])
metrics = {
    'Silhouette\n(↑ better)': silhouette,
    'DB Index\n(↓ better)': 1 / (davies_bouldin + 0.1),  # Inverse for visualization
    'CH Score\n(↑ better)': calinski / 1000,  # Scale down
    'Avg Conf\n(↑ better)': max_probs.mean()
}
bars = ax2.barh(list(metrics.keys()), list(metrics.values()), 
                color=['green', 'orange', 'blue', 'purple'], 
                edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Normalized Score', fontsize=10, fontweight='bold')
ax2.set_title('Quality Metrics', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
for i, (k, v) in enumerate(metrics.items()):
    ax2.text(v, i, f' {v:.3f}', va='center', fontsize=9, fontweight='bold')

# 3. Confidence heatmap
ax3 = fig.add_subplot(gs[1, :])
conf_by_cluster = [probabilities[clusters_df['gmm_cluster'] == i, i] for i in range(num_clusters)]
positions = []
for i, conf in enumerate(conf_by_cluster):
    positions.extend([i] * len(conf))
confidence_flat = np.concatenate(conf_by_cluster)
hist = ax3.hist2d(positions, confidence_flat, bins=[num_clusters, 30], 
                  cmap='YlOrRd', cmin=1)
plt.colorbar(hist[3], ax=ax3, label='Frequency')
ax3.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax3.set_ylabel('Assignment Confidence', fontsize=11, fontweight='bold')
ax3.set_title('Confidence Density by Cluster', fontsize=13, fontweight='bold')
ax3.set_xticks(np.arange(num_clusters) + 0.5)
ax3.set_xticklabels(range(num_clusters))

# 4. Low confidence analysis
ax4 = fig.add_subplot(gs[2, 0])
low_conf_threshold = 0.5
low_conf_counts = [(probabilities[clusters_df['gmm_cluster'] == i, i] < low_conf_threshold).sum() 
                   for i in range(num_clusters)]
ax4.bar(range(num_clusters), low_conf_counts, color='coral', 
       edgecolor='black', linewidth=1.5, alpha=0.7)
ax4.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.set_title(f'Low Confidence Assignments\n(< {low_conf_threshold})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
for i, count in enumerate(low_conf_counts):
    if count > 0:
        ax4.text(i, count, f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 5. BIC/AIC comparison
ax5 = fig.add_subplot(gs[2, 1])
ic_metrics = {'BIC': metadata['bic'], 'AIC': metadata['aic']}
bars = ax5.bar(ic_metrics.keys(), ic_metrics.values(), 
              color=['steelblue', 'teal'], edgecolor='black', linewidth=1.5, alpha=0.7)
ax5.set_ylabel('Score (lower is better)', fontsize=10, fontweight='bold')
ax5.set_title('Information Criteria', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
for bar, (k, v) in zip(bars, ic_metrics.items()):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Model summary text
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
summary_text = f"""
MODEL SUMMARY

Clusters: {num_clusters}
SKUs: {num_skus:,}
Features: {num_features_used}/{num_features_original}

QUALITY SCORES
Silhouette: {silhouette:.4f}
Davies-Bouldin: {davies_bouldin:.4f}
Calinski-Harabasz: {calinski:.0f}

CONFIDENCE
Mean: {max_probs.mean():.4f}
Min: {max_probs.min():.4f}
Max: {max_probs.max():.4f}

IMBALANCE
Ratio: {metadata['imbalance_ratio']:.2f}
"""
ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('GMM Clustering Quality Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(RESULTS_PATH + 'gmm_quality_dashboard.png', dpi=150, bbox_inches='tight')
print(f"📊 Quality dashboard saved: {RESULTS_PATH}gmm_quality_dashboard.png")
plt.close()

# -----------------------
# Save Enhanced Results
# -----------------------
# Save enhanced cluster assignments
enhanced_clusters = pd.DataFrame({
    'sku_id': feature_df['sku_id'],
    'gmm_cluster': clusters_df['gmm_cluster'],
    'cluster_probability': max_probs,
    'second_best_cluster': np.argsort(probabilities, axis=1)[:, -2],
    'second_best_probability': np.sort(probabilities, axis=1)[:, -2],
    'confidence_gap': max_probs - np.sort(probabilities, axis=1)[:, -2]
})
enhanced_clusters.to_csv(OUTPUT_PATH + 'gmm_clusters_enhanced.csv', index=False)

# Save detailed cluster statistics
cluster_stats.to_csv(RESULTS_PATH + 'gmm_cluster_stats_detailed.csv', index=False)

# Save comprehensive quality metrics
quality_metrics = {
    'model_info': {
        'num_clusters': int(num_clusters),
        'num_skus': int(num_skus),
        'num_features_original': int(num_features_original),
        'num_features_used': int(num_features_used),
        'use_pca': bool(use_pca),
        'scaler_type': metadata['scaler_type'],
        'covariance_type': metadata['best_covariance'],
        'n_init': metadata['best_n_init'],
        'converged': bool(metadata['converged'])
    },
    'information_criteria': {
        'bic_score': float(metadata['bic']),
        'aic_score': float(metadata['aic'])
    },
    'quality_metrics': {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_score': float(calinski)
    },
    'confidence_metrics': {
        'mean_confidence': float(max_probs.mean()),
        'median_confidence': float(np.median(max_probs)),
        'min_confidence': float(max_probs.min()),
        'max_confidence': float(max_probs.max()),
        'std_confidence': float(max_probs.std()),
        'low_confidence_count': int((max_probs < 0.5).sum()),
        'low_confidence_percentage': float((max_probs < 0.5).sum() / num_skus * 100)
    },
    'cluster_balance': {
        'imbalance_ratio': float(metadata['imbalance_ratio']),
        'cluster_sizes': cluster_stats['size'].tolist(),
        'cluster_percentages': cluster_stats['percentage'].tolist()
    },
    'pca_info': {
        'explained_variance_ratio_2d': [float(x) for x in pca_viz.explained_variance_ratio_],
        'cumulative_variance_2d': float(sum(pca_viz.explained_variance_ratio_))
    }
}

if use_pca:
    quality_metrics['pca_info']['model_components'] = int(pca_model.n_components_)
    quality_metrics['pca_info']['model_explained_variance'] = [float(x) for x in pca_model.explained_variance_ratio_]
    quality_metrics['pca_info']['model_cumulative_variance'] = float(sum(pca_model.explained_variance_ratio_))

with open(RESULTS_PATH + 'gmm_quality_metrics_comprehensive.json', 'w') as f:
    json.dump(quality_metrics, f, indent=2)

print(f"\n💾 Enhanced clusters saved: {OUTPUT_PATH}gmm_clusters_enhanced.csv")
print(f"💾 Detailed stats saved: {RESULTS_PATH}gmm_cluster_stats_detailed.csv")
print(f"💾 Comprehensive metrics saved: {RESULTS_PATH}gmm_quality_metrics_comprehensive.json")

# -----------------------
# Summary Report
# -----------------------
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

# Interpret silhouette score
if silhouette > 0.5:
    silhouette_interpretation = "✅ Excellent - Strong cluster structure"
elif silhouette > 0.3:
    silhouette_interpretation = "✅ Good - Reasonable cluster structure"
elif silhouette > 0.1:
    silhouette_interpretation = "⚠️  Fair - Weak cluster structure"
else:
    silhouette_interpretation = "❌ Poor - Clusters may overlap significantly"

print(f"\n🎯 Silhouette Score: {silhouette:.4f}")
print(f"   {silhouette_interpretation}")

# Davies-Bouldin interpretation
if davies_bouldin < 0.5:
    db_interpretation = "✅ Excellent - Tight, well-separated clusters"
elif davies_bouldin < 1.0:
    db_interpretation = "✅ Good - Decent cluster separation"
else:
    db_interpretation = "⚠️  Fair - Some cluster overlap"

print(f"\n🎯 Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"   {db_interpretation}")

# Check for imbalanced clusters
max_cluster_pct = cluster_stats['percentage'].max()
min_cluster_pct = cluster_stats['percentage'].min()
imbalance_ratio = max_cluster_pct / min_cluster_pct if min_cluster_pct > 0 else float('inf')

if imbalance_ratio > 5:
    print(f"\n⚠️  WARNING: Highly imbalanced clusters detected")
    print(f"   Largest: {max_cluster_pct:.1f}%, Smallest: {min_cluster_pct:.1f}%")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}")
elif imbalance_ratio > 3:
    print(f"\n⚠️  Moderate cluster imbalance")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}")
else:
    print(f"\n✅ Well-balanced clusters")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}")

# Check confidence
mean_conf = max_probs.mean()
low_conf_pct = (max_probs < 0.5).sum() / num_skus * 100

if mean_conf > 0.8:
    print(f"\n✅ High average confidence ({mean_conf:.3f})")
    print(f"   Clear cluster assignments for most SKUs")
elif mean_conf > 0.6:
    print(f"\n⚠️  Moderate confidence ({mean_conf:.3f})")
    print(f"   Some ambiguous assignments")
else:
    print(f"\n❌ Low confidence ({mean_conf:.3f})")
    print(f"   Many ambiguous assignments")

if low_conf_pct > 10:
    print(f"   ⚠️  {low_conf_pct:.1f}% of SKUs have confidence < 0.5")

# Model convergence check
if not metadata['converged']:
    print(f"\n⚠️  WARNING: Model did not converge")
    print(f"   Consider increasing max_iter or simplifying the model")

# Feature reduction summary
if use_pca:
    variance_retained = sum(pca_model.explained_variance_ratio_)
    print(f"\n📊 PCA Dimensionality Reduction:")
    print(f"   {num_features_original} → {num_features_used} features")
    print(f"   Retained {variance_retained*100:.1f}% of variance")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print(f"\nGenerated {sum(1 for f in os.listdir(RESULTS_PATH) if f.startswith('gmm_'))} visualization files")
print(f"Results saved to: {RESULTS_PATH}")