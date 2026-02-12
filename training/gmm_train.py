import pandas as pd
import numpy as np
import os
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURE_PATH = "../synthetic_data_scripts/derived_data/gmm_features.csv"
OUTPUT_PATH = "../synthetic_data_scripts/derived_data/"
MODEL_PATH = "../models/gmm/"

os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------
# Load features
# -----------------------
df = pd.read_csv(FEATURE_PATH)

sku_ids = df["sku_id"]
X = df.drop(columns=["sku_id"])

# -----------------------
# Scale features
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, MODEL_PATH + "gmm_scaler.pkl")

# -----------------------
# Select best K using BIC
# -----------------------
bic_scores = []
k_range = range(2, 9)

for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))

best_k = k_range[np.argmin(bic_scores)]
print(f"Optimal clusters (BIC): {best_k}")

# -----------------------
# Train final GMM
# -----------------------
final_gmm = GaussianMixture(
    n_components=best_k,
    covariance_type="full",
    random_state=42
)
final_gmm.fit(X_scaled)

joblib.dump(final_gmm, MODEL_PATH + "gmm_model.pkl")

# -----------------------
# Assign clusters
# -----------------------
clusters = final_gmm.predict(X_scaled)

result = pd.DataFrame({
    "sku_id": sku_ids,
    "gmm_cluster": clusters
})

result.to_csv(OUTPUT_PATH + "gmm_clusters.csv", index=False)

print("✅ GMM training completed successfully")
