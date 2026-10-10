import pandas as pd
import numpy as np
import os
import sys

WINDOW_SIZE = 30

def generate_lstm_features(input_path, output_path):

    demand = pd.read_csv(os.path.join(input_path, "demand_history.csv"))

    sequences = []
    targets = []

    for sku_id, group in demand.groupby("sku_id"):
        group = group.sort_values("date")
        series = group["demand_qty"].values

        for i in range(len(series) - WINDOW_SIZE):
            sequences.append(series[i:i + WINDOW_SIZE])
            targets.append(series[i + WINDOW_SIZE])

    X = np.array(sequences)
    y = np.array(targets)

    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, "lstm_sequences.npy"), X)
    np.save(os.path.join(output_path, "test_targets.npy"), y)

    print("✅ LSTM features generated")


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_lstm_features(input_path, output_path)