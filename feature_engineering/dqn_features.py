import pandas as pd
import numpy as np
import os
import sys

def generate_dqn_features(input_path, output_path):

    df = pd.read_csv(os.path.join(input_path, "dqn_episodes.csv"))

    df["distance"] = np.sqrt(
        (df["current_x"] - df["next_x"])**2 +
        (df["current_y"] - df["next_y"])**2
    )

    df["reward"] = -df["distance"]

    os.makedirs(output_path, exist_ok=True)

    df.to_csv(
        os.path.join(output_path, "dqn_features.csv"),
        index=False
    )

    print("✅ DQN features generated")


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_dqn_features(input_path, output_path)