import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt

MODEL_PATH = "../models/gmm/"
OUTPUT_PATH = "../synthetic_data_scripts/derived_data/"
RESULTS_PATH = "../resultimage/"

os.makedirs(RESULTS_PATH, exist_ok=True)

# -----------------------
# Load Model and Data
# -----------------------
print("="*70)
print("GMM CLUSTERING - EVALUATION")
print("="*70)

# Load model files
if not os.path.exists(MODEL_PATH + "gmm_model.pkl"):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("Please train the model first using gmm_train.py")
    exit(1)

gmm_model = joblib.load(MODEL_PATH + "gmm_model.pkl")
scaler = joblib.load(MODEL_PATH + "gmm_scaler.pkl")

# Load cluster assignments
clusters_df = pd.read_csv(OUTPUT_PATH + "gmm_clusters.csv")

# Load original features
feature_df = pd.read_csv("../synthetic_data_scripts/derived_data/gmm_features.csv")
X = feature_df.drop(columns=["sku_id"])
X_scaled = scaler.transform(X)

# Get model info
num_clusters = gmm_model.n_components
num_skus = len(clusters_df)
num_features = X.shape[1]

print(f"\nModel Information:")
print(f"  Number of clusters: {num_clusters}")
print(f"  Number of SKUs: {num_skus}")
print(f"  Number of features: {num_features}")
print(f"  BIC Score: {gmm_model.bic(X_scaled):.2f}")
print(f"  AIC Score: {gmm_model.aic(X_scaled):.2f}")

# -----------------------
# Calculate Statistics
# -----------------------
probabilities = gmm_model.predict_proba(X_scaled)

cluster_stats = pd.DataFrame({
    'cluster': range(num_clusters),
    'size': [(clusters_df['gmm_cluster'] == i).sum() for i in range(num_clusters)],
    'percentage': [(clusters_df['gmm_cluster'] == i).sum() / num_skus * 100 for i in range(num_clusters)],
    'avg_confidence': [probabilities[clusters_df['gmm_cluster'] == i, i].mean() for i in range(num_clusters)]
})

print("\n" + "="*70)
print("CLUSTER STATISTICS")
print("="*70)
print(cluster_stats.to_string(index=False))

# -----------------------
# Quality Metrics
# -----------------------
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

silhouette = silhouette_score(X_scaled, clusters_df['gmm_cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, clusters_df['gmm_cluster'])
calinski = calinski_harabasz_score(X_scaled, clusters_df['gmm_cluster'])

print("\n" + "="*70)
print("CLUSTERING QUALITY METRICS")
print("="*70)
print(f"Silhouette Score:        {silhouette:.4f}  (higher is better, range: -1 to 1)")
print(f"Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower is better)")
print(f"Calinski-Harabasz Score: {calinski:.4f}  (higher is better)")
print(f"Mean Confidence:         {probabilities.max(axis=1).mean():.4f}")

# -----------------------
# Visualization 1: Cluster Distribution
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cluster size distribution
ax1.bar(cluster_stats['cluster'], cluster_stats['size'], 
        color='steelblue', edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Cluster', fontsize=12)
ax1.set_ylabel('Number of SKUs', fontsize=12)
ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for i, row in cluster_stats.iterrows():
    ax1.text(row['cluster'], row['size'] + max(cluster_stats['size'])*0.02, 
            f"{row['percentage']:.1f}%", ha='center', fontsize=10)

# Pie chart
ax2.pie(cluster_stats['size'], labels=[f"Cluster {i}" for i in range(num_clusters)],
        autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax2.set_title('Cluster Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'gmm_cluster_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Cluster distribution saved: {RESULTS_PATH}gmm_cluster_distribution.png")

# -----------------------
# Visualization 2: Confidence Analysis
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Overall confidence distribution
confidence_scores = probabilities.max(axis=1)
ax1.hist(confidence_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(confidence_scores.mean(), color='r', linestyle='--', linewidth=2, 
           label=f"Mean: {confidence_scores.mean():.3f}")
ax1.set_xlabel('Assignment Confidence', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Cluster Assignment Confidence Distribution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Confidence by cluster (boxplot)
cluster_confidences = [probabilities[clusters_df['gmm_cluster'] == i, i].tolist() 
                       for i in range(num_clusters)]
bp = ax2.boxplot(cluster_confidences, labels=range(num_clusters), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
    patch.set_alpha(0.7)
ax2.set_xlabel('Cluster', fontsize=12)
ax2.set_ylabel('Assignment Confidence', fontsize=12)
ax2.set_title('Confidence Distribution by Cluster', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'gmm_confidence_analysis.png', dpi=150, bbox_inches='tight')
print(f"📊 Confidence analysis saved: {RESULTS_PATH}gmm_confidence_analysis.png")

# -----------------------
# Visualization 3: PCA Projection
# -----------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=clusters_df['gmm_cluster'], 
                     cmap='tab10', 
                     s=50, 
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
plt.title('GMM Clusters - PCA 2D Projection', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'gmm_pca_projection.png', dpi=150, bbox_inches='tight')
print(f"📊 PCA projection saved: {RESULTS_PATH}gmm_pca_projection.png")

# -----------------------
# Visualization 4: Feature Profiles (if <= 10 features)
# -----------------------
feature_names = list(X.columns)

if len(feature_names) <= 10:
    fig, axes = plt.subplots(1, num_clusters, figsize=(5*num_clusters, 4))
    if num_clusters == 1:
        axes = [axes]
    
    for i in range(num_clusters):
        cluster_mask = clusters_df['gmm_cluster'] == i
        cluster_features = X[cluster_mask]
        
        feature_means = cluster_features.mean()
        
        axes[i].barh(feature_names, feature_means, color='coral', edgecolor='black')
        axes[i].set_xlabel('Mean Value', fontsize=10)
        axes[i].set_title(f'Cluster {i}\n({cluster_stats.iloc[i]["size"]} SKUs)', 
                         fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + 'gmm_feature_profiles.png', dpi=150, bbox_inches='tight')
    print(f"📊 Feature profiles saved: {RESULTS_PATH}gmm_feature_profiles.png")
else:
    print(f"⚠️  Skipping feature profiles (too many features: {len(feature_names)})")

# -----------------------
# Save Results
# -----------------------
# Save enhanced cluster assignments
enhanced_clusters = pd.DataFrame({
    'sku_id': feature_df['sku_id'],
    'gmm_cluster': clusters_df['gmm_cluster'],
    'cluster_probability': probabilities.max(axis=1)
})
enhanced_clusters.to_csv(OUTPUT_PATH + 'gmm_clusters_with_confidence.csv', index=False)

# Save cluster statistics
cluster_stats.to_csv(RESULTS_PATH + 'gmm_cluster_stats.csv', index=False)

# Save quality metrics
quality_metrics = {
    'num_clusters': int(num_clusters),
    'num_skus': int(num_skus),
    'num_features': int(num_features),
    'bic_score': float(gmm_model.bic(X_scaled)),
    'aic_score': float(gmm_model.aic(X_scaled)),
    'silhouette_score': float(silhouette),
    'davies_bouldin_index': float(davies_bouldin),
    'calinski_harabasz_score': float(calinski),
    'mean_confidence': float(probabilities.max(axis=1).mean()),
    'min_confidence': float(probabilities.max(axis=1).min()),
    'max_confidence': float(probabilities.max(axis=1).max()),
    'pca_variance_explained': [float(x) for x in pca.explained_variance_ratio_]
}

with open(RESULTS_PATH + 'gmm_quality_metrics.json', 'w') as f:
    json.dump(quality_metrics, f, indent=2)

print(f"\n💾 Enhanced clusters saved: {OUTPUT_PATH}gmm_clusters_with_confidence.csv")
print(f"💾 Cluster stats saved: {RESULTS_PATH}gmm_cluster_stats.csv")
print(f"💾 Quality metrics saved: {RESULTS_PATH}gmm_quality_metrics.json")

# -----------------------
# Summary Report
# -----------------------
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

# Interpret silhouette score
if silhouette > 0.5:
    silhouette_interpretation = "Excellent - Strong cluster structure"
elif silhouette > 0.3:
    silhouette_interpretation = "Good - Reasonable cluster structure"
elif silhouette > 0.1:
    silhouette_interpretation = "Fair - Weak cluster structure"
else:
    silhouette_interpretation = "Poor - Clusters may overlap significantly"

print(f"\nSilhouette Score: {silhouette:.4f} - {silhouette_interpretation}")

# Check for imbalanced clusters
max_cluster_pct = cluster_stats['percentage'].max()
min_cluster_pct = cluster_stats['percentage'].min()
if max_cluster_pct / min_cluster_pct > 5:
    print(f"⚠️  Warning: Highly imbalanced clusters detected")
    print(f"   Largest: {max_cluster_pct:.1f}%, Smallest: {min_cluster_pct:.1f}%")

# Check confidence
if quality_metrics['mean_confidence'] > 0.8:
    print(f"✅ High average confidence ({quality_metrics['mean_confidence']:.3f}) - Clear cluster assignments")
elif quality_metrics['mean_confidence'] > 0.6:
    print(f"⚠️  Moderate confidence ({quality_metrics['mean_confidence']:.3f}) - Some ambiguous assignments")
else:
    print(f"❌ Low confidence ({quality_metrics['mean_confidence']:.3f}) - Many ambiguous assignments")

print("\n" + "="*70)
print("✅ Evaluation complete!")
print("="*70)