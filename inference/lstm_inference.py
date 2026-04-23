import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import pandas as pd
import torch
from model_architecture.lstm import LSTMModel

dataset_path = sys.argv[1]
result_path = sys.argv[2]

os.makedirs(result_path, exist_ok=True)

SEQ_LEN = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load model
# ------------------------
model = LSTMModel(input_dim=2).to(DEVICE)
model.load_state_dict(torch.load("models/lstm/lstm_demand.pt", map_location=DEVICE))
model.eval()

# ------------------------
# Load demand history
# ------------------------
demand = pd.read_csv(os.path.join(dataset_path, "demand_history.csv"))

forecast_results = []

# ------------------------
# Forecast per SKU
# ------------------------
for sku in demand["sku_id"].unique():

    sku_df = demand[demand["sku_id"] == sku]

    values = sku_df[["demand_qty", "promotion_flag"]].values.astype(float)

    if len(values) < SEQ_LEN:
        continue

    # shape → [1, SEQ_LEN, 2]
    seq = torch.tensor(values[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(seq).item()

    forecast_results.append({
        "sku_id": sku,
        "predicted_demand": pred
    })

# ------------------------
# Save results
# ------------------------
forecast_df = pd.DataFrame(forecast_results)

forecast_df.to_csv(
    os.path.join(result_path, "demand_forecast.csv"),
    index=False
)

print("LSTM inference completed")