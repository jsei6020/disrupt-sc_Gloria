import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path

####missing pre 12 condition
#############################################

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"
OUT_FILE = "vulnerability_direct_indirect_avg_ratio_by_region.csv"


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: AVERAGE INDIRECT/DIRECT LOSS RATIO BY REGION")
    print("=" * 80)

    # --- 1. Load Runs Mapping ---
    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True].copy()
    print(f"  Found {len(runs)} valid simulations")

    if "folder_name" not in runs.columns or "region_sector" not in runs.columns:
        raise ValueError("Expected 'folder_name' and 'region_sector' columns in runs file")

    folder_to_disruption = dict(zip(runs["folder_name"], runs["region_sector"]))

    # --- 2. Load Region Name Mapping ---
    print("\n[2/5] Loading region table...")
    try:
        regions_df = pd.read_csv(REGIONS_TABLE_FILE)
        if "Region_acronyms" not in regions_df.columns or "Region_names" not in regions_df.columns:
            raise ValueError("Expected 'Region_acronyms' and 'Region_names' columns in regions_table.csv")

        region_name_map = dict(
            zip(
                regions_df["Region_acronyms"].astype(str),
                regions_df["Region_names"].astype(str)
            )
        )
        print(f"  Loaded region names for {len(region_name_map)} regions")
    except Exception as e:
        print(f"  ERROR loading {REGIONS_TABLE_FILE}: {e}")
        return

    # --- 3. Process loss files and collect per-run ratios ---
    print("\n[3/5] Processing loss files...")

    run_region_ratios = []
    processed_count = 0

    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        disrupted_region_sector = folder_to_disruption.get(folder)

        if not disrupted_region_sector:
            continue

        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")

            try:
                df_loss = pd.read_csv(loss_path)

                required_cols = {"region", "sector", "loss"}
                if not required_cols.issubset(df_loss.columns):
                    del df_loss
                    processed_count += 1
                    continue

                # Direct if affected sector matches disrupted sector in this run
                df_loss["is_direct"] = df_loss["sector"] == disrupted_region_sector

                # Sum losses within this run by affected region and direct/indirect
                grouped = (
                    df_loss.groupby(["region", "is_direct"], as_index=False)["loss"]
                    .sum()
                )

                # Reshape to one row per region within this run
                pivot = (
                    grouped.pivot(index="region", columns="is_direct", values="loss")
                    .fillna(0.0)
                    .reset_index()
                )

                # Handle missing True/False columns safely
                pivot["direct_loss"] = pivot.get(True, 0.0)
                pivot["indirect_loss"] = pivot.get(False, 0.0)

                # Compute run-level ratio for each affected region
                pivot["indirect_direct_ratio"] = np.where(
                    pivot["direct_loss"] > 0,
                    pivot["indirect_loss"] / pivot["direct_loss"],
                    np.nan
                )

                pivot["run_folder"] = folder

                run_region_ratios.append(
                    pivot[["run_folder", "region", "direct_loss", "indirect_loss", "indirect_direct_ratio"]]
                )

                del df_loss
                del grouped
                del pivot
                gc.collect()
                processed_count += 1

            except Exception as e:
                print(f"\n  ERROR processing {folder}: {e}")
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")

    print(f"\n  Completed processing {processed_count} simulation files")

    if not run_region_ratios:
        print("\nERROR: No run-level ratios could be calculated.")
        return

    # --- 4. Average run-level ratios by region ---
    print("\n[4/5] Aggregating average ratios by region...")

    df_all = pd.concat(run_region_ratios, ignore_index=True)

    df_results = (
        df_all.groupby("region", as_index=False)
        .agg(
            avg_direct_loss=("direct_loss", "mean"),
            avg_indirect_loss=("indirect_loss", "mean"),
            avg_indirect_direct_ratio=("indirect_direct_ratio", "mean"),
            valid_run_count=("indirect_direct_ratio", lambda x: x.notna().sum()),
            total_run_count=("run_folder", "nunique")
        )
        .rename(columns={"region": "iso"})
    )

    df_results["region_name"] = df_results["iso"].astype(str).map(region_name_map)
    df_results = df_results[
        ["iso", "region_name", "avg_direct_loss", "avg_indirect_loss",
         "avg_indirect_direct_ratio", "valid_run_count", "total_run_count"]
    ]

    df_results = df_results.sort_values(
        "avg_indirect_direct_ratio",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    # --- 5. Save output ---
    print("\n[5/5] Saving results...")
    df_results.to_csv(OUT_FILE, index=False)

    print(f"\n{'=' * 80}")
    print(f"DONE! Results saved to {OUT_FILE}")
    print(f"{'=' * 80}")

    print("\n📊 TOP 10 REGIONS BY AVERAGE INDIRECT/DIRECT LOSS RATIO:")
    print(
        df_results[
            ["iso", "region_name", "avg_indirect_direct_ratio",
             "avg_direct_loss", "avg_indirect_loss", "valid_run_count"]
        ]
        .head(10)
        .to_string(index=True)
    )


if __name__ == "__main__":
    main()