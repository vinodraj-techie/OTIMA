import pandas as pd
import json
import os

DATA_PATH = "../synthetic_data_scripts/derived_data/"
MODEL_PATH = "../models/abc/"

os.makedirs(MODEL_PATH, exist_ok=True)

def train_abc():
    df = pd.read_csv(DATA_PATH + "abc_results.csv")

    rules = {
        "method": "ABC Classification",
        "description": "Classification based on cumulative annual consumption value",
        "thresholds": {
            "A": 0.7,
            "B": 0.9,
            "C": 1.0
        },
        "num_skus": int(df.shape[0])
    }

    with open(MODEL_PATH + "abc_rules.json", "w") as f:
        json.dump(rules, f, indent=4)

    print("✅ ABC rules saved successfully")

if __name__ == "__main__":
    train_abc()
