import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "../synthetic_data_scripts/data/demand_history.csv"
MODEL_PATH = "../models/lstm/"
os.makedirs(MODEL_PATH, exist_ok=True)

SEQ_LEN = 14
BATCH_SIZE = 64
EPOCHS = 25
LR = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["sku_id", "date"])

# -----------------------------
# Build sequences per SKU
# -----------------------------
X = []
y = []

demand_scaler = MinMaxScaler()

for sku, g in df.groupby("sku_id"):

    values = g[["demand_qty", "promotion_flag"]].values.astype(float)

    # scale only demand, keep promotion as-is
    demand_scaled = demand_scaler.fit_transform(
        values[:, 0].reshape(-1, 1)
    )

    promo = values[:, 1].reshape(-1, 1)

    features = np.hstack([demand_scaled, promo])

    for i in range(len(features) - SEQ_LEN):
        X.append(features[i:i+SEQ_LEN])
        y.append(demand_scaled[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

# -----------------------------
# Train / Val split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Torch datasets
# -----------------------------
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -----------------------------
# LSTM Model
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

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# -----------------------------
# Training
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
    )

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), os.path.join(MODEL_PATH, "lstm_demand.pt"))

print("✅ LSTM training completed")
