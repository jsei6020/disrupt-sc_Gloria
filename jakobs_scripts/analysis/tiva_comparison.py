
#!/usr/bin/env python3
"""
Compute final demand import dependency from an OECD TiVA CSV.

Definition:
final_demand_import_dependency(country) =
    imported_final_demand / total_final_demand

Where:
- total_final_demand = sum by FINAL_DEMAND_AREA
- imported_final_demand = sum by FINAL_DEMAND_AREA where
  VALUE_ADDED_SOURCE_AREA != FINAL_DEMAND_AREA

Filter:
- keep only rows where FINAL_DEMAND_AREA is a 3-letter ISO-like code
- keep only rows where VALUE_ADDED_SOURCE_AREA is a 3-letter ISO-like code
"""

import pandas as pd
import numpy as np

# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/Databases/Tiva/tiva.csv"
OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/Databases/Tiva/fd_import_dependency_oecd_tiva.csv"

# Set this to the numeric value column in your CSV
VALUE_COLUMN = "OBS_VALUE"
FD_NAME_COLUMN = "Final demand area"   # <- existing full-name column in your CSV

def main():
    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "FINAL_DEMAND_AREA",
        "VALUE_ADDED_SOURCE_AREA",
        FD_NAME_COLUMN,
        VALUE_COLUMN
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["FINAL_DEMAND_AREA"] = df["FINAL_DEMAND_AREA"].astype(str).str.strip()
    df["VALUE_ADDED_SOURCE_AREA"] = df["VALUE_ADDED_SOURCE_AREA"].astype(str).str.strip()

    iso_mask = (
        df["FINAL_DEMAND_AREA"].str.fullmatch(r"[A-Z]{3}") &
        df["VALUE_ADDED_SOURCE_AREA"].str.fullmatch(r"[A-Z]{3}")
    )
    df = df.loc[iso_mask].copy()

    df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce").fillna(0)

    total_fd = (
        df.groupby("FINAL_DEMAND_AREA", as_index=False)
          .agg(
              **{
                  "Final demand area": (FD_NAME_COLUMN, "first"),
                  "total_final_demand": (VALUE_COLUMN, "sum"),
              }
          )
    )

    imported_fd = (
        df.loc[df["VALUE_ADDED_SOURCE_AREA"] != df["FINAL_DEMAND_AREA"]]
          .groupby("FINAL_DEMAND_AREA", as_index=False)
          .agg(imported_final_demand=(VALUE_COLUMN, "sum"))
    )

    result = total_fd.merge(imported_fd, on="FINAL_DEMAND_AREA", how="left")
    result["imported_final_demand"] = result["imported_final_demand"].fillna(0)

    result["final_demand_import_dependency"] = np.where(
        result["total_final_demand"] != 0,
        result["imported_final_demand"] / result["total_final_demand"],
        np.nan
    )

    result["final_demand_import_dependency_pct"] = (
        result["final_demand_import_dependency"] * 100
    ).round(2)

    result = result.rename(columns={"FINAL_DEMAND_AREA": "region"})

    result = result[
        [
            "region",
            "Final demand area",
            "total_final_demand",
            "imported_final_demand",
            "final_demand_import_dependency",
            "final_demand_import_dependency_pct",
        ]
    ]

    result = result.sort_values(
        by=["final_demand_import_dependency", "region"],
        ascending=[False, True],
        na_position="last"
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()