import pandas as pd
import numpy as np

DATA_PATH = "../../synthetic_data_scripts/derived_data/"

def generate_dqn_features():
    df = pd.read_csv(DATA_PATH + "dqn_episodes.csv")

    df["distance"] = np.sqrt(
        (df["current_x"] - df["next_x"])**2 +
        (df["current_y"] - df["next_y"])**2
    )

    df["reward"] = -df["distance"]

    df.to_csv(DATA_PATH + "dqn_features.csv", index=False)
    print("DQN features generated")

if __name__ == "__main__":
    generate_dqn_features()
