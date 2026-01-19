#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/supply_concentrations/region_sector_pairs_above_1pct2.csv"
OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/supply_concentrations/country_export_concentration.csv"
# ========================


def main():
    # Load CSV
    df = pd.read_csv(INPUT_PATH)

    # Sanity checks
    required_cols = {"region_sector", "supply_share_exports"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Extract ISO3 code = first 3 characters of region_sector [web:55]
    df["iso3"] = df["region_sector"].astype(str).str[:3]

    # Group by iso3 and compute:
    # - occurrences (count of rows)
    # - total_supply_share_exports (sum of supply_share_exports) [web:52][web:64]
    grouped = (
        df.groupby("iso3")
        .agg(
            occurrences=("region_sector", "count"),
            total_supply_share_exports=("supply_share_exports", "sum"),
        )
        .reset_index()
    )

    # Sort descending by total_supply_share_exports
    grouped = grouped.sort_values(
        "total_supply_share_exports", ascending=False
    ).reset_index(drop=True)

    # Export to CSV
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)

    print(f"Written summary to: {out_path}")
    print("Top 10 rows:")
    print(grouped.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
