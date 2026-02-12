# gmm_train_optimized.py
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import zscore
warnings.filterwarnings('ignore')

FEATURE_PATH = "../synthetic_data_scripts/derived_data/gmm_features.csv"
OUTPUT_PATH = "../synthetic_data_scripts/derived_data/"
MODEL_PATH = "../models/gmm2/"

os.makedirs(MODEL_PATH, exist_ok=True)

print("="*70)
print("OPTIMIZED GMM CLUSTERING")
print("="*70)

# -----------------------
# Load features
# -----------------------
df = pd.read_csv(FEATURE_PATH)
sku_ids = df["sku_id"]
X = df.drop(columns=["sku_id"])

print(f"\nLoaded {len(X)} SKUs with {X.shape[1]} features")
print(f"Features: {list(X.columns)}")

# -----------------------
# Data Quality Check & Preprocessing
# -----------------------
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Check for missing values
if X.isnull().any().any():
    print("⚠️  Missing values detected. Filling with median...")
    X = X.fillna(X.median())

# Check for infinite values
if np.isinf(X.values).any():
    print("⚠️  Infinite values detected. Replacing with max/min...")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

# Remove zero-variance features
zero_var_cols = X.columns[X.var() == 0]
if len(zero_var_cols) > 0:
    print(f"⚠️  Removing {len(zero_var_cols)} zero-variance features: {list(zero_var_cols)}")
    X = X.drop(columns=zero_var_cols)

# Outlier detection and handling
print("\nOutlier Detection:")
z_scores = np.abs(zscore(X))
outlier_mask = (z_scores > 3).any(axis=1)
num_outliers = outlier_mask.sum()
print(f"  Found {num_outliers} outliers (z-score > 3)")

if num_outliers > 0 and num_outliers < len(X) * 0.1:  # Less than 10% outliers
    print(f"  Action: Capping extreme values")
    # Cap outliers at 3 standard deviations
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        X[col] = X[col].clip(lower=mean - 3*std, upper=mean + 3*std)
else:
    print(f"  Action: Keeping all data (robust scaling will handle)")

# Feature correlation analysis
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
highly_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

if highly_corr:
    print(f"\n⚠️  High correlation detected. Removing {len(highly_corr)} redundant features: {highly_corr}")
    X = X.drop(columns=highly_corr)

print(f"\n✅ Final feature set: {X.shape[1]} features")

# -----------------------
# Feature Scaling (Test Multiple Methods)
# -----------------------
print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler()  # Better for data with outliers
}

# Try both scalers and pick the best
best_scaler = None
best_score = -np.inf

for scaler_name, scaler in scalers.items():
    X_scaled_test = scaler.fit_transform(X)
    
    # Quick test with k=3
    gmm_test = GaussianMixture(n_components=3, covariance_type='full', 
                                random_state=42, n_init=3)
    labels_test = gmm_test.fit_predict(X_scaled_test)
    
    if len(np.unique(labels_test)) > 1:
        sil_score = silhouette_score(X_scaled_test, labels_test)
        print(f"{scaler_name:15} → Silhouette: {sil_score:.4f}")
        
        if sil_score > best_score:
            best_score = sil_score
            best_scaler = scaler
    else:
        print(f"{scaler_name:15} → Failed (single cluster)")

scaler = best_scaler
X_scaled = scaler.fit_transform(X)
print(f"\n✅ Selected: {type(scaler).__name__}")

joblib.dump(scaler, MODEL_PATH + "gmm_scaler.pkl")

# -----------------------
# Dimensionality Reduction (Optional but Recommended)
# -----------------------
print("\n" + "="*70)
print("DIMENSIONALITY ANALYSIS")
print("="*70)

if X_scaled.shape[1] > 10:
    pca = PCA()
    pca.fit(X_scaled)
    
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_components_99 = np.argmax(cumsum_var >= 0.99) + 1
    
    print(f"Original dimensions: {X_scaled.shape[1]}")
    print(f"Dimensions for 95% variance: {n_components_95}")
    print(f"Dimensions for 99% variance: {n_components_99}")
    
    # Use PCA if significant reduction possible
    if n_components_95 < X_scaled.shape[1] * 0.7:
        print(f"\n✅ Applying PCA (keeping {n_components_95} components)")
        pca = PCA(n_components=n_components_95, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        joblib.dump(pca, MODEL_PATH + "gmm_pca.pkl")
        use_pca = True
    else:
        print(f"\n⚠️  PCA not beneficial, using original features")
        use_pca = False
else:
    print(f"Feature count ({X_scaled.shape[1]}) is reasonable, skipping PCA")
    use_pca = False

# -----------------------
# Model Selection: Test Multiple Configurations
# -----------------------
print("\n" + "="*70)
print("MODEL SELECTION")
print("="*70)

k_range = range(2, min(11, len(X) // 50))  # Up to 10 clusters or data size limit
covariance_types = ['full', 'tied', 'diag', 'spherical']
n_init_values = [10, 20]  # Multiple initializations

results = []

print(f"Testing {len(k_range)} cluster counts × {len(covariance_types)} covariance types × {len(n_init_values)} initializations...")
print(f"Total: {len(k_range) * len(covariance_types) * len(n_init_values)} configurations\n")

for k in k_range:
    for cov_type in covariance_types:
        for n_init in n_init_values:
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    n_init=n_init,
                    random_state=42,
                    max_iter=200,
                    tol=1e-4
                )
                
                labels = gmm.fit_predict(X_scaled)
                
                # Check if valid clustering
                n_clusters = len(np.unique(labels))
                if n_clusters < 2:
                    continue
                
                # Compute metrics
                bic = gmm.bic(X_scaled)
                aic = gmm.aic(X_scaled)
                
                # Only compute expensive metrics for promising models
                if len(results) == 0 or bic < sorted([r['bic'] for r in results])[-3:][0]:
                    sil = silhouette_score(X_scaled, labels)
                    db = davies_bouldin_score(X_scaled, labels)
                    ch = calinski_harabasz_score(X_scaled, labels)
                else:
                    sil = db = ch = None
                
                results.append({
                    'k': k,
                    'cov_type': cov_type,
                    'n_init': n_init,
                    'bic': bic,
                    'aic': aic,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch,
                    'n_clusters': n_clusters,
                    'converged': gmm.converged_,
                    'model': gmm,
                    'labels': labels
                })
                
            except Exception as e:
                continue

if not results:
    print("❌ No valid models found. Using default k=3, full covariance")
    final_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    final_gmm.fit(X_scaled)
    clusters = final_gmm.predict(X_scaled)
    best_k = 3
else:
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Calculate composite score (normalize and combine metrics)
    # Lower BIC/AIC/DB is better, higher Silhouette/CH is better
    
    # Filter to only converged models
    results_df = results_df[results_df['converged'] == True]
    
    if len(results_df) == 0:
        print("⚠️  No converged models, using best non-converged")
        results_df = pd.DataFrame(results)
    
    # Compute only for models with full metrics
    scored_results = results_df[results_df['silhouette'].notna()].copy()
    
    if len(scored_results) > 0:
        # Normalize metrics (min-max scaling)
        scored_results['bic_norm'] = 1 - (scored_results['bic'] - scored_results['bic'].min()) / (scored_results['bic'].max() - scored_results['bic'].min() + 1e-10)
        scored_results['sil_norm'] = (scored_results['silhouette'] - scored_results['silhouette'].min()) / (scored_results['silhouette'].max() - scored_results['silhouette'].min() + 1e-10)
        scored_results['db_norm'] = 1 - (scored_results['davies_bouldin'] - scored_results['davies_bouldin'].min()) / (scored_results['davies_bouldin'].max() - scored_results['davies_bouldin'].min() + 1e-10)
        scored_results['ch_norm'] = (scored_results['calinski_harabasz'] - scored_results['calinski_harabasz'].min()) / (scored_results['calinski_harabasz'].max() - scored_results['calinski_harabasz'].min() + 1e-10)
        
        # Weighted composite score
        scored_results['composite_score'] = (
            0.3 * scored_results['bic_norm'] +
            0.3 * scored_results['sil_norm'] +
            0.2 * scored_results['db_norm'] +
            0.2 * scored_results['ch_norm']
        )
        
        # Sort by composite score
        scored_results = scored_results.sort_values('composite_score', ascending=False)
        
        # Display top 5 models
        print("Top 5 Models by Composite Score:")
        print("-"*70)
        for idx, row in scored_results.head(5).iterrows():
            print(f"k={row['k']}, cov={row['cov_type']}, n_init={row['n_init']}: "
                  f"Score={row['composite_score']:.4f}, Sil={row['silhouette']:.4f}, BIC={row['bic']:.0f}")
        
        # Select best model
        best_idx = scored_results.index[0]
        best_result = results_df.loc[best_idx]
    else:
        # Fall back to BIC only
        print("⚠️  Using BIC-only selection (no quality metrics available)")
        results_df = results_df.sort_values('bic')
        best_result = results_df.iloc[0]
    
    best_k = best_result['k']
    best_cov = best_result['cov_type']
    best_n_init = best_result['n_init']
    final_gmm = best_result['model']
    clusters = best_result['labels']
    
    print("\n" + "="*70)
    print("BEST MODEL SELECTED")
    print("="*70)
    print(f"Clusters (k):        {best_k}")
    print(f"Covariance type:     {best_cov}")
    print(f"Initializations:     {best_n_init}")
    print(f"Converged:           {best_result['converged']}")
    print(f"BIC:                 {best_result['bic']:.2f}")
    print(f"AIC:                 {best_result['aic']:.2f}")
    if best_result['silhouette'] is not None:
        print(f"Silhouette:          {best_result['silhouette']:.4f}")
        print(f"Davies-Bouldin:      {best_result['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz:   {best_result['calinski_harabasz']:.0f}")

# -----------------------
# Save Model
# -----------------------
joblib.dump(final_gmm, MODEL_PATH + "gmm_model.pkl")

# -----------------------
# Assign Clusters with Probabilities
# -----------------------
probabilities = final_gmm.predict_proba(X_scaled)
max_probs = probabilities.max(axis=1)

result = pd.DataFrame({
    "sku_id": sku_ids,
    "gmm_cluster": clusters,
    "confidence": max_probs
})

result.to_csv(OUTPUT_PATH + "gmm_clusters.csv", index=False)

# -----------------------
# Cluster Statistics
# -----------------------
print("\n" + "="*70)
print("CLUSTER STATISTICS")
print("="*70)

cluster_stats = []
for i in range(best_k):
    cluster_mask = clusters == i
    cluster_size = cluster_mask.sum()
    cluster_pct = (cluster_size / len(clusters)) * 100
    avg_confidence = max_probs[cluster_mask].mean()
    
    cluster_stats.append({
        'cluster': i,
        'size': cluster_size,
        'percentage': cluster_pct,
        'avg_confidence': avg_confidence
    })
    
    print(f"Cluster {i}: {cluster_size:4d} SKUs ({cluster_pct:5.1f}%) | "
          f"Confidence: {avg_confidence:.3f}")

cluster_stats_df = pd.DataFrame(cluster_stats)
cluster_stats_df.to_csv(OUTPUT_PATH + "gmm_cluster_stats.csv", index=False)

# Check for imbalance
max_pct = cluster_stats_df['percentage'].max()
min_pct = cluster_stats_df['percentage'].min()
imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')

if imbalance_ratio > 5:
    print(f"\n⚠️  High cluster imbalance (ratio: {imbalance_ratio:.2f})")
elif imbalance_ratio > 3:
    print(f"\n⚠️  Moderate cluster imbalance (ratio: {imbalance_ratio:.2f})")
else:
    print(f"\n✅ Well-balanced clusters (ratio: {imbalance_ratio:.2f})")

# -----------------------
# Save Metadata
# -----------------------
import json

metadata = {
    'num_skus': len(sku_ids),
    'num_features_original': len(df.columns) - 1,
    'num_features_final': X_scaled.shape[1],
    'features_used': list(X.columns), 
    'use_pca': use_pca,
    'scaler_type': type(scaler).__name__,
    'best_k': int(best_k),
    'best_covariance': best_cov if 'best_cov' in locals() else 'full',
    'best_n_init': int(best_n_init) if 'best_n_init' in locals() else 10,
    'converged': bool(final_gmm.converged_),
    'bic': float(final_gmm.bic(X_scaled)),
    'aic': float(final_gmm.aic(X_scaled)),
    'imbalance_ratio': float(imbalance_ratio),
    'avg_confidence': float(max_probs.mean()),
    'min_confidence': float(max_probs.min())
}

with open(MODEL_PATH + "gmm_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ GMM training completed successfully")
print(f"   Model: {MODEL_PATH}gmm_model.pkl")
print(f"   Scaler: {MODEL_PATH}gmm_scaler.pkl")
if use_pca:
    print(f"   PCA: {MODEL_PATH}gmm_pca.pkl")
print(f"   Clusters: {OUTPUT_PATH}gmm_clusters.csv")
print(f"   Stats: {OUTPUT_PATH}gmm_cluster_stats.csv")
print(f"   Metadata: {MODEL_PATH}gmm_metadata.json")