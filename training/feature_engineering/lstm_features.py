import pandas as pd
import numpy as np

DATA_PATH = "../../synthetic_data_scripts/data/"
OUTPUT_PATH = "../../synthetic_data_scripts/derived_data/forecasting/"

WINDOW_SIZE = 30

def generate_lstm_features():
    demand = pd.read_csv(DATA_PATH + "demand_history.csv")

    sequences = []
    targets = []

    for sku_id, group in demand.groupby("sku_id"):
        series = group.sort_values("date")["demand_qty"].values

        for i in range(len(series) - WINDOW_SIZE):
            sequences.append(series[i:i+WINDOW_SIZE])
            targets.append(series[i+WINDOW_SIZE])

    X = np.array(sequences)
    y = np.array(targets)

    np.save(OUTPUT_PATH + "lstm_sequences.npy", X)
    np.save(OUTPUT_PATH + "test_targets.npy", y)

    print("LSTM features generated")

if __name__ == "__main__":
    generate_lstm_features()
