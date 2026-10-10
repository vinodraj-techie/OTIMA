import os
import subprocess
from datetime import datetime
import sys


def run_pipeline(dataset_path):

    print("\n===== OPTIMA PIPELINE STARTED =====")

    # create run id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    derived_path = f"derived_data/run_{run_id}"
    results_path = f"results/run_{run_id}"

    os.makedirs(derived_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    print("\nStep 1: Feature Engineering")

    subprocess.run([
        sys.executable,
        "feature_engineering/abc_features.py",
        dataset_path,
        derived_path
    ])

    subprocess.run([
       sys.executable,
        "feature_engineering/gmm_features.py",
        dataset_path,
        derived_path
    ])

    subprocess.run([
       sys.executable,
        "feature_engineering/slap_features.py",
        dataset_path,
        derived_path
    ])

    subprocess.run([
       sys.executable,
        "feature_engineering/dqn_features.py",
        dataset_path,
        derived_path
    ])
    subprocess.run([
       sys.executable,
        "feature_engineering/lstm_features.py",
        dataset_path,
        derived_path
    ])

    print("\nStep 2: Running Model Inference")

    subprocess.run([
       sys.executable,
        "inference/abc_inference.py",
        dataset_path,
        results_path
    ])

    subprocess.run([
       sys.executable,
        "inference/gmm_inference.py",
        derived_path,
        results_path
    ])

    subprocess.run([
       sys.executable,
        "inference/lstm_inference.py",
        dataset_path,
        results_path
    ])

    subprocess.run([
       sys.executable,
        "inference/slap_inference2.py",
        derived_path,
        results_path
    ])

    subprocess.run([
       sys.executable,
        "inference/dqn_inference.py",
        dataset_path,
        results_path
    ])

    print("\n===== PIPELINE FINISHED =====")
    print("Results saved at:", results_path)

    return results_path


if __name__ == "__main__":

    dataset = input("Enter dataset folder path: ")

    if not os.path.exists(dataset):
        print("Dataset path not found")
        exit()

    run_pipeline(dataset)