import pandas as pd

DATA_PATH = "../../synthetic_data_scripts/data/"
OUTPUT_PATH = "../../synthetic_data_scripts/derived_data/"

def generate_abc_features():
    df = pd.read_csv(DATA_PATH + "sku_master.csv")

    df["annual_consumption_value"] = df["unit_cost"] * df["annual_demand"]

    df = df.sort_values("annual_consumption_value", ascending=False)
    cumulative = df["annual_consumption_value"].cumsum() / df["annual_consumption_value"].sum()

    df["abc_class"] = cumulative.apply(
        lambda x: "A" if x <= 0.7 else ("B" if x <= 0.9 else "C")
    )

    df[[
        "sku_id",
        "unit_cost",
        "annual_demand",
        "annual_consumption_value",
        "abc_class"
    ]].to_csv(OUTPUT_PATH + "abc_results.csv", index=False)

    print("ABC features generated")

if __name__ == "__main__":
    generate_abc_features()
