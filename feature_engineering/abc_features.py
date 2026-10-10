import pandas as pd
import os
import sys

def generate_abc_features(input_path, output_path):

    # -------------------------------
    # Load SKU master data
    # -------------------------------
    df = pd.read_csv(os.path.join(input_path, "sku_master.csv"))

    # -------------------------------
    # Calculate annual consumption value
    # -------------------------------
    df["annual_consumption_value"] = df["unit_cost"] * df["annual_demand"]

    # -------------------------------
    # Sort by value (descending)
    # -------------------------------
    df = df.sort_values("annual_consumption_value", ascending=False)

    cumulative = df["annual_consumption_value"].cumsum() / df["annual_consumption_value"].sum()

    # -------------------------------
    # Assign ABC classes
    # -------------------------------
    df["abc_class"] = cumulative.apply(
        lambda x: "A" if x <= 0.7 else ("B" if x <= 0.9 else "C")
    )

    # -------------------------------
    # Ensure output directory exists
    # -------------------------------
    os.makedirs(output_path, exist_ok=True)

    # -------------------------------
    # Save results
    # -------------------------------
    df[[
        "sku_id",
        "unit_cost",
        "annual_demand",
        "annual_consumption_value",
        "abc_class"
    ]].to_csv(
        os.path.join(output_path, "abc_results.csv"),
        index=False
    )

    print("✅ ABC classification features generated")


if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    generate_abc_features(input_path, output_path)