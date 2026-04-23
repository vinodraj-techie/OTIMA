import sys
import pandas as pd
import joblib
import os

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

model = joblib.load("models/gmm2/gmm_model.pkl")


features = pd.read_csv(f"{dataset_path}/gmm_features.csv")

X_raw = features.drop(columns=["sku_id"])

# Critical Fix: Load and apply the scaler/pca to inference data
scaler_path = "models/gmm2/gmm_scaler.pkl"
pca_path = "models/gmm2/gmm_pca.pkl"
metadata_path = "models/gmm2/gmm_metadata.json"

X = X_raw.copy()

# Dynamically align to the exact features kept by the training algorithm, ignoring any correlated drops
import json
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    X = X[meta['features_used']]

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)

if os.path.exists(pca_path):
    pca = joblib.load(pca_path)
    X = pca.transform(X)

clusters = model.predict(X)
confidence = model.predict_proba(X).max(axis=1)

output = pd.DataFrame({
    "sku_id": features["sku_id"],
    "gmm_cluster": clusters,
    "confidence": confidence
})

output.to_csv(f"{result_path}/gmm_clusters.csv", index=False)

print("GMM inference completed")