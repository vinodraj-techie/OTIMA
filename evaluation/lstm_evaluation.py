import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import os

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "../synthetic_data_scripts/data/demand_history.csv"
MODEL_PATH = "../models/lstm/"
RESULTS_PATH = "../resultimage/lstm/"
os.makedirs(RESULTS_PATH, exist_ok=True)

SEQ_LEN = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("LSTM DEMAND FORECASTING - EVALUATION")
print("="*70)
print(f"Using device: {DEVICE}")

# -----------------------------
# Load Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel().to(DEVICE)

try:
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "lstm_demand.pt"), 
                                     map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# -----------------------------
# Load and Prepare Data
# -----------------------------
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["sku_id", "date"])

print(f"\nDataset Information:")
print(f"  Total records: {len(df):,}")
print(f"  Unique SKUs: {df['sku_id'].nunique():,}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# -----------------------------
# Build Sequences and Make Predictions
# -----------------------------
predictions_list = []
actuals_list = []
sku_ids_list = []
dates_list = []

demand_scaler = MinMaxScaler()

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

sku_count = 0
total_skus = df['sku_id'].nunique()

for sku, g in df.groupby("sku_id"):
    sku_count += 1
    if sku_count % 100 == 0:
        print(f"Processing SKU {sku_count}/{total_skus}...")
    
    values = g[["demand_qty", "promotion_flag"]].values.astype(float)
    
    if len(values) <= SEQ_LEN:
        continue  # Skip SKUs with insufficient data
    
    # Scale demand
    demand_scaled = demand_scaler.fit_transform(values[:, 0].reshape(-1, 1))
    promo = values[:, 1].reshape(-1, 1)
    features = np.hstack([demand_scaled, promo])
    
    # Generate sequences and predictions
    for i in range(len(features) - SEQ_LEN):
        sequence = features[i:i+SEQ_LEN]
        actual_scaled = demand_scaled[i+SEQ_LEN]
        
        # Predict
        with torch.no_grad():
            x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_scaled = model(x_tensor).cpu().numpy()[0, 0]
        
        # Inverse transform
        pred = demand_scaler.inverse_transform([[pred_scaled]])[0, 0]
        actual = demand_scaler.inverse_transform(actual_scaled.reshape(1, -1))[0, 0]
        
        predictions_list.append(pred)
        actuals_list.append(actual)
        sku_ids_list.append(sku)
        dates_list.append(g.iloc[i+SEQ_LEN]["date"])

print(f"✅ Generated {len(predictions_list):,} predictions")

# -----------------------------
# Calculate Metrics
# -----------------------------
predictions = np.array(predictions_list)
actuals = np.array(actuals_list)

mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

# MAPE with proper handling of zero/near-zero demands
# Only calculate MAPE for demands > threshold to avoid division issues
mape_threshold = 1.0  # Only include demands >= 1
valid_mape_mask = actuals >= mape_threshold
if valid_mape_mask.sum() > 0:
    mape = np.mean(np.abs((actuals[valid_mape_mask] - predictions[valid_mape_mask]) / actuals[valid_mape_mask])) * 100
    mape_coverage = valid_mape_mask.sum() / len(actuals) * 100
else:
    mape = np.nan
    mape_coverage = 0

print("\n" + "="*70)
print("OVERALL PERFORMANCE METRICS")
print("="*70)
print(f"MSE:          {mse:.4f}")
print(f"RMSE:         {rmse:.4f}")
print(f"MAE:          {mae:.4f}")
print(f"R² Score:     {r2:.4f}")
if not np.isnan(mape):
    print(f"MAPE:         {mape:.2f}% (calculated on {mape_coverage:.1f}% of data with demand ≥ {mape_threshold})")
else:
    print(f"MAPE:         N/A (no demands ≥ {mape_threshold})")

# -----------------------------
# Per-SKU Performance
# -----------------------------
results_df = pd.DataFrame({
    'sku_id': sku_ids_list,
    'date': dates_list,
    'actual': actuals_list,
    'predicted': predictions_list,
    'error': np.array(predictions_list) - np.array(actuals_list),
    'abs_error': np.abs(np.array(predictions_list) - np.array(actuals_list))
})

# Calculate percentage error only for demands >= threshold
results_df['pct_error'] = np.where(
    results_df['actual'] >= mape_threshold,
    np.abs((results_df['actual'] - results_df['predicted']) / results_df['actual']) * 100,
    np.nan
)

sku_metrics = results_df.groupby('sku_id').agg({
    'actual': 'mean',
    'predicted': 'mean',
    'abs_error': 'mean',
    'pct_error': lambda x: np.nanmean(x)  # Use nanmean to ignore NaN values
}).rename(columns={
    'actual': 'avg_actual_demand',
    'predicted': 'avg_predicted_demand',
    'abs_error': 'mae',
    'pct_error': 'mape'
}).reset_index()

# Add coverage information (% of predictions with valid MAPE)
sku_coverage = results_df.groupby('sku_id')['pct_error'].apply(
    lambda x: (~x.isna()).sum() / len(x) * 100
).reset_index()
sku_coverage.columns = ['sku_id', 'mape_coverage_pct']
sku_metrics = sku_metrics.merge(sku_coverage, on='sku_id')

# Calculate per-SKU R²
sku_r2 = []
for sku in sku_metrics['sku_id']:
    sku_data = results_df[results_df['sku_id'] == sku]
    if len(sku_data) > 1:
        r2_sku = r2_score(sku_data['actual'], sku_data['predicted'])
        sku_r2.append(r2_sku)
    else:
        sku_r2.append(np.nan)

sku_metrics['r2_score'] = sku_r2

print("\n" + "="*70)
print("PER-SKU PERFORMANCE SUMMARY")
print("="*70)
print(f"Average MAE per SKU:    {sku_metrics['mae'].mean():.4f}")
mape_valid_skus = sku_metrics['mape'].notna()
if mape_valid_skus.sum() > 0:
    print(f"Average MAPE per SKU:   {sku_metrics.loc[mape_valid_skus, 'mape'].mean():.2f}% (across {mape_valid_skus.sum()} SKUs)")
else:
    print(f"Average MAPE per SKU:   N/A (no valid MAPE calculations)")
print(f"Average R² per SKU:     {sku_metrics['r2_score'].mean():.4f}")
print(f"\nTop 5 Best Performing SKUs (by R²):")
print(sku_metrics.nlargest(5, 'r2_score')[['sku_id', 'mae', 'mape', 'r2_score']].to_string(index=False))
print(f"\nTop 5 Worst Performing SKUs (by R²):")
print(sku_metrics.nsmallest(5, 'r2_score')[['sku_id', 'mae', 'mape', 'r2_score']].to_string(index=False))

# -----------------------------
# Visualization 1: Actual vs Predicted
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
max_val = max(actuals.max(), predictions.max())
ax1.scatter(actuals, predictions, alpha=0.3, s=10, color='steelblue')
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Demand', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Demand', fontsize=12, fontweight='bold')
ax1.set_title('Actual vs Predicted Demand', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Residual plot
residuals = predictions - actuals
ax2.scatter(predictions, residuals, alpha=0.3, s=10, color='coral')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Demand', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'lstm_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Actual vs Predicted plot saved: {RESULTS_PATH}lstm_actual_vs_predicted.png")
plt.close()

# -----------------------------
# Visualization 2: Error Distribution
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Error histogram
ax1.hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(residuals.mean(), color='r', linestyle='--', linewidth=2, 
           label=f'Mean: {residuals.mean():.2f}')
ax1.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
ax1.set_xlabel('Prediction Error', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Percentage error histogram
valid_pct_errors = results_df['pct_error'].dropna()
if len(valid_pct_errors) > 0:
    ax2.hist(valid_pct_errors.clip(0, 100), bins=50, color='lightcoral', 
             edgecolor='black', alpha=0.7)
    ax2.axvline(valid_pct_errors.median(), color='r', linestyle='--', linewidth=2,
               label=f'Median: {valid_pct_errors.median():.2f}%')
    ax2.set_xlabel('Absolute Percentage Error (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Percentage Error Distribution\n({len(valid_pct_errors):,}/{len(results_df):,} predictions)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
else:
    ax2.text(0.5, 0.5, 'No valid MAPE data\n(all demands < threshold)', 
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'lstm_error_distribution.png', dpi=150, bbox_inches='tight')
print(f"📊 Error distribution plot saved: {RESULTS_PATH}lstm_error_distribution.png")
plt.close()

# -----------------------------
# Visualization 3: Time Series Sample (3 random SKUs)
# -----------------------------
sample_skus = np.random.choice(results_df['sku_id'].unique(), size=min(3, len(results_df['sku_id'].unique())), replace=False)

fig, axes = plt.subplots(len(sample_skus), 1, figsize=(14, 4*len(sample_skus)))
if len(sample_skus) == 1:
    axes = [axes]

for idx, sku in enumerate(sample_skus):
    sku_data = results_df[results_df['sku_id'] == sku].sort_values('date')
    
    axes[idx].plot(sku_data['date'], sku_data['actual'], 
                   label='Actual', linewidth=2, color='blue', marker='o', markersize=3)
    axes[idx].plot(sku_data['date'], sku_data['predicted'], 
                   label='Predicted', linewidth=2, color='red', linestyle='--', marker='x', markersize=3)
    
    axes[idx].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Demand', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'SKU: {sku} (MAE: {sku_data["abs_error"].mean():.2f}, R²: {r2_score(sku_data["actual"], sku_data["predicted"]):.3f})', 
                       fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3, linestyle='--')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'lstm_timeseries_sample.png', dpi=150, bbox_inches='tight')
print(f"📊 Time series sample saved: {RESULTS_PATH}lstm_timeseries_sample.png")
plt.close()

# -----------------------------
# Visualization 4: Performance Dashboard
# -----------------------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Metrics summary
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
mape_text = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
metrics_text = f"""
OVERALL METRICS

MSE:      {mse:.4f}
RMSE:     {rmse:.4f}
MAE:      {mae:.4f}
R² Score: {r2:.4f}
MAPE:     {mape_text}

PREDICTIONS

Total:    {len(predictions):,}
SKUs:     {len(sku_metrics):,}
"""
ax1.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# 2. MAE distribution by SKU
ax2 = fig.add_subplot(gs[0, 1:])
ax2.hist(sku_metrics['mae'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(sku_metrics['mae'].mean(), color='r', linestyle='--', linewidth=2,
           label=f'Mean: {sku_metrics["mae"].mean():.2f}')
ax2.set_xlabel('MAE', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of SKUs', fontsize=11, fontweight='bold')
ax2.set_title('MAE Distribution Across SKUs', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# 3. R² distribution by SKU
ax3 = fig.add_subplot(gs[1, :])
valid_r2 = sku_metrics['r2_score'].dropna()
ax3.hist(valid_r2, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.axvline(valid_r2.mean(), color='r', linestyle='--', linewidth=2,
           label=f'Mean: {valid_r2.mean():.3f}')
ax3.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of SKUs', fontsize=11, fontweight='bold')
ax3.set_title('R² Score Distribution Across SKUs', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# 4. Error by demand level
ax4 = fig.add_subplot(gs[2, 0])
demand_bins = pd.qcut(results_df['actual'], q=5, duplicates='drop')
error_by_bin = results_df.groupby(demand_bins, observed=False)['abs_error'].mean()
ax4.bar(range(len(error_by_bin)), error_by_bin.values, color='coral', 
        edgecolor='black', alpha=0.7)
ax4.set_xlabel('Demand Quintile', fontsize=10, fontweight='bold')
ax4.set_ylabel('Mean Absolute Error', fontsize=10, fontweight='bold')
ax4.set_title('Error by Demand Level', fontsize=12, fontweight='bold')
ax4.set_xticks(range(len(error_by_bin)))
ax4.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'][:len(error_by_bin)], fontsize=9)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# 5. MAPE distribution
ax5 = fig.add_subplot(gs[2, 1])
valid_mape_skus = sku_metrics['mape'].dropna()
if len(valid_mape_skus) > 0:
    ax5.hist(valid_mape_skus.clip(0, 100), bins=30, color='orange', 
            edgecolor='black', alpha=0.7)
    ax5.axvline(valid_mape_skus.median(), color='r', linestyle='--', linewidth=2,
               label=f'Median: {valid_mape_skus.median():.1f}%')
    ax5.set_xlabel('MAPE (%)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Number of SKUs', fontsize=10, fontweight='bold')
    ax5.set_title(f'MAPE Distribution\n({len(valid_mape_skus)} SKUs)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
else:
    ax5.text(0.5, 0.5, 'No valid MAPE data', ha='center', va='center', 
            transform=ax5.transAxes, fontsize=10)

# 6. Performance categories
ax6 = fig.add_subplot(gs[2, 2])
r2_categories = pd.cut(valid_r2, bins=[-np.inf, 0, 0.5, 0.7, 0.9, 1.0], 
                       labels=['Poor\n(<0)', 'Fair\n(0-0.5)', 'Good\n(0.5-0.7)', 
                              'Very Good\n(0.7-0.9)', 'Excellent\n(0.9-1.0)'])
category_counts = r2_categories.value_counts().sort_index()
colors_cat = ['red', 'orange', 'yellow', 'lightgreen', 'green']
ax6.bar(range(len(category_counts)), category_counts.values, 
        color=colors_cat[:len(category_counts)], edgecolor='black', alpha=0.7)
ax6.set_xlabel('Performance Category', fontsize=10, fontweight='bold')
ax6.set_ylabel('Number of SKUs', fontsize=10, fontweight='bold')
ax6.set_title('Performance Categories', fontsize=12, fontweight='bold')
ax6.set_xticks(range(len(category_counts)))
ax6.set_xticklabels(category_counts.index, fontsize=8)
ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
for i, v in enumerate(category_counts.values):
    ax6.text(i, v, str(v), ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.suptitle('LSTM Demand Forecasting - Performance Dashboard', 
            fontsize=16, fontweight='bold', y=0.995)
plt.savefig(RESULTS_PATH + 'lstm_performance_dashboard.png', dpi=150, bbox_inches='tight')
print(f"📊 Performance dashboard saved: {RESULTS_PATH}lstm_performance_dashboard.png")
plt.close()

# -----------------------------
# Save Results
# -----------------------------
# Save all predictions
results_df.to_csv(RESULTS_PATH + 'lstm_predictions.csv', index=False)
print(f"\n💾 Predictions saved: {RESULTS_PATH}lstm_predictions.csv")

# Save per-SKU metrics
sku_metrics.to_csv(RESULTS_PATH + 'lstm_sku_metrics.csv', index=False)
print(f"💾 SKU metrics saved: {RESULTS_PATH}lstm_sku_metrics.csv")

# Save summary metrics
summary_metrics = {
    'overall_metrics': {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'mape': float(mape) if not np.isnan(mape) else None,
        'mape_coverage_pct': float(mape_coverage) if not np.isnan(mape) else 0,
        'mape_threshold': float(mape_threshold)
    },
    'per_sku_averages': {
        'avg_mae': float(sku_metrics['mae'].mean()),
        'avg_mape': float(sku_metrics['mape'].mean()) if sku_metrics['mape'].notna().sum() > 0 else None,
        'avg_r2': float(sku_metrics['r2_score'].mean()),
        'skus_with_valid_mape': int(sku_metrics['mape'].notna().sum())
    },
    'data_info': {
        'total_predictions': int(len(predictions)),
        'num_skus': int(len(sku_metrics)),
        'sequence_length': int(SEQ_LEN)
    },
    'performance_categories': {
        'excellent': int((valid_r2 >= 0.9).sum()),
        'very_good': int(((valid_r2 >= 0.7) & (valid_r2 < 0.9)).sum()),
        'good': int(((valid_r2 >= 0.5) & (valid_r2 < 0.7)).sum()),
        'fair': int(((valid_r2 >= 0) & (valid_r2 < 0.5)).sum()),
        'poor': int((valid_r2 < 0).sum())
    }
}

with open(RESULTS_PATH + 'lstm_evaluation_summary.json', 'w') as f:
    json.dump(summary_metrics, f, indent=2)
print(f"💾 Summary metrics saved: {RESULTS_PATH}lstm_evaluation_summary.json")

# -----------------------------
# Final Summary
# -----------------------------
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

# Interpret R² score
if r2 > 0.9:
    r2_interpretation = "✅ Excellent - Model explains >90% of variance"
elif r2 > 0.7:
    r2_interpretation = "✅ Very Good - Strong predictive power"
elif r2 > 0.5:
    r2_interpretation = "✅ Good - Reasonable predictive power"
elif r2 > 0:
    r2_interpretation = "⚠️  Fair - Limited predictive power"
else:
    r2_interpretation = "❌ Poor - Model worse than baseline"

print(f"\n🎯 R² Score: {r2:.4f}")
print(f"   {r2_interpretation}")

# Interpret MAPE
if not np.isnan(mape):
    if mape < 10:
        mape_interpretation = "✅ Excellent - Very accurate forecasts"
    elif mape < 20:
        mape_interpretation = "✅ Good - Acceptable forecast accuracy"
    elif mape < 50:
        mape_interpretation = "⚠️  Fair - Moderate forecast errors"
    else:
        mape_interpretation = "❌ Poor - High forecast errors"
    
    print(f"\n🎯 MAPE: {mape:.2f}% (on {mape_coverage:.1f}% of predictions with demand ≥ {mape_threshold})")
    print(f"   {mape_interpretation}")
else:
    print(f"\n🎯 MAPE: Not calculated (no demands ≥ {mape_threshold})")
    print(f"   Consider using MAE or RMSE as primary metrics for low-demand items")

# SKU performance summary
excellent_skus = (valid_r2 >= 0.9).sum()
poor_skus = (valid_r2 < 0).sum()

print(f"\n📊 SKU Performance Breakdown:")
print(f"   {excellent_skus}/{len(valid_r2)} SKUs ({excellent_skus/len(valid_r2)*100:.1f}%) have excellent R² (≥0.9)")
if poor_skus > 0:
    print(f"   ⚠️  {poor_skus}/{len(valid_r2)} SKUs ({poor_skus/len(valid_r2)*100:.1f}%) have poor R² (<0)")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
print(f"\nGenerated {sum(1 for f in os.listdir(RESULTS_PATH) if f.startswith('lstm_'))} output files")
print(f"Results saved to: {RESULTS_PATH}")
