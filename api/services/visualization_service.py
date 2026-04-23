import pandas as pd
import matplotlib
matplotlib.use('Agg') # MUST be called before pyplot
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
        df["gmm_cluster"].value_counts().plot(kind="bar")

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
    # DQN route visualization (Map of SLAP & Route)
    # --------------------
    dqn_file = os.path.join(results_path, "optimized_route.csv")

    if os.path.exists(dqn_file):

        df = pd.read_csv(dqn_file)

        # Locate dataset layout automatically based on run_id
        run_name = os.path.basename(results_path) 
        uploads_folder = os.path.join("uploads", run_name)
        
        layout_df = None
        if os.path.exists(uploads_folder):
            for item in os.listdir(uploads_folder):
                potential_dataset = os.path.join(uploads_folder, item)
                if os.path.isdir(potential_dataset):
                    layout_path = os.path.join(potential_dataset, "warehouse_layout.csv")
                    if os.path.exists(layout_path):
                        layout_df = pd.read_csv(layout_path)
                        break

        # Plot map physical layout!
        if layout_df is not None and "route" in df.columns:
            plt.figure(figsize=(12, 10))
            
            # Plot all bins (empty background map)
            plt.scatter(layout_df['x_coord'], layout_df['y_coord'], c='gainsboro', marker='s', s=120, label='Empty Bins')
            
            # Plot SLAP utilized storage
            if os.path.exists(slap_file):
                slap_storage = pd.read_csv(slap_file)
                if 'assigned_bin' in slap_storage.columns:
                    used_bins = slap_storage['assigned_bin'].unique()
                    used_layout = layout_df[layout_df['bin_id'].isin(used_bins)]
                    plt.scatter(used_layout['x_coord'], used_layout['y_coord'], c='royalblue', marker='s', s=120, alpha=0.8, label='SLAP Allocated Storage')
            
            # Plot DQN optimized route
            route_bins = df['route'].tolist()
            route_coords = layout_df[layout_df['bin_id'].isin(route_bins)].set_index('bin_id').reindex(route_bins)
            
            # Draw lines bridging picked items
            plt.plot(route_coords['x_coord'], route_coords['y_coord'], marker='o', markersize=6, c='crimson', linestyle='--', linewidth=2.5, zorder=3, label='DQN Picking Route')
            
            # Plot Start position star
            if len(route_coords) > 0:
                plt.scatter([route_coords['x_coord'].iloc[0]], [route_coords['y_coord'].iloc[0]], c='gold', edgecolors='black', marker='*', s=450, zorder=5, label='Picking Start Point')
            
            plt.title("Warehouse Top-down Map: SLAP Allocations & DQN Route", fontsize=16)
            plt.xlabel("X Coordinate", fontsize=12)
            plt.ylabel("Y Coordinate", fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_path, "dqn_route_distance.png"))
            plt.close()
        elif "route" in df.columns:
            # Fallback simple route chart if dataset map cannot be found
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(df)), df["route"], marker='o', linestyle='-', color='indigo')
            plt.title("DQN Route Sequence (Warehouse Map missing)")
            plt.xlabel("Step Number")
            plt.ylabel("Bin ID")
            plt.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_path, "dqn_route_distance.png"))
            plt.close()

    return charts_path