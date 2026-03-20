import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations(results_path):

    charts_path = os.path.join(results_path, "charts")
    os.makedirs(charts_path, exist_ok=True)

    sns.set(style="whitegrid")

    # --------------------
    # ABC visualization
    # --------------------
    abc_file = os.path.join(results_path, "abc_results.csv")

    if os.path.exists(abc_file):

        df = pd.read_csv(abc_file)

        plt.figure()
        df["abc_class"].value_counts().plot(kind="bar")

        plt.title("ABC Classification Distribution")
        plt.xlabel("ABC Class")
        plt.ylabel("Number of SKUs")

        plt.savefig(os.path.join(charts_path, "abc_distribution.png"))
        plt.close()

    # --------------------
    # GMM clusters
    # --------------------
    gmm_file = os.path.join(results_path, "gmm_clusters.csv")

    if os.path.exists(gmm_file):

        df = pd.read_csv(gmm_file)

        plt.figure()
        df["cluster"].value_counts().plot(kind="bar")

        plt.title("GMM Cluster Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of SKUs")

        plt.savefig(os.path.join(charts_path, "gmm_clusters.png"))
        plt.close()

    # --------------------
    # Demand forecast
    # --------------------
    forecast_file = os.path.join(results_path, "demand_forecast.csv")

    if os.path.exists(forecast_file):

        df = pd.read_csv(forecast_file)

        plt.figure()

        sns.histplot(df["predicted_demand"], bins=30)

        plt.title("Demand Forecast Distribution")
        plt.xlabel("Predicted Demand")

        plt.savefig(os.path.join(charts_path, "forecast_distribution.png"))
        plt.close()

    # --------------------
    # SLAP visualization
    # --------------------
    slap_file = os.path.join(results_path, "optimized_storage.csv")

    if os.path.exists(slap_file):

        df = pd.read_csv(slap_file)

        plt.figure(figsize=(10,5))

        df["assigned_bin"].value_counts().head(20).plot(kind="bar")

        plt.title("Top Used Warehouse Bins (SLAP)")
        plt.xlabel("Bin ID")
        plt.ylabel("Number of SKUs")

        plt.savefig(os.path.join(charts_path, "slap_bin_usage.png"))
        plt.close()

    # --------------------
    # DQN route visualization
    # --------------------
    dqn_file = os.path.join(results_path, "dqn_route.csv")

    if os.path.exists(dqn_file):

        df = pd.read_csv(dqn_file)

        if "step" in df.columns and "distance" in df.columns:

            plt.figure()

            plt.plot(df["step"], df["distance"])

            plt.title("DQN Route Distance per Step")
            plt.xlabel("Step")
            plt.ylabel("Distance")

            plt.savefig(os.path.join(charts_path, "dqn_route_distance.png"))
            plt.close()

    return charts_path