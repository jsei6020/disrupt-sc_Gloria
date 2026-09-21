import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_all.csv"
TOTAL_OUTPUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/total_region_output.csv"

# New output file focused on self-damage shares
OUT_FILE = "/home/user/Documents/University/Master Thesis/threat_analysis/t-sh3.csv"

# thousands USD -> trillions USD
SCALE_TO_TRILLIONS = 1e-9


def get_banning_region_from_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'region' column exists representing the original export-banning country.
    Priority:
      1. region_sector -> split before first underscore
      2. region
      3. folder_name -> split before first underscore (fallback, only if naming allows)
    """
    runs = runs.copy()

    if "region_sector" in runs.columns:
        runs["region"] = runs["region_sector"].astype(str).str.split("_", n=1).str[0]
    elif "region" in runs.columns:
        runs["region"] = runs["region"].astype(str)
    elif "folder_name" in runs.columns:
        runs["region"] = runs["folder_name"].astype(str).str.split("_", n=1).str[0]
    else:
        raise ValueError("RUNS_FILE must contain 'region_sector', 'region', or inferable 'folder_name'.")

    return runs


def select_top3_runs_per_region(runs: pd.DataFrame) -> pd.DataFrame:
    """
    From the runs DataFrame, read loss_summary.csv for each run,
    compute households_loss_trillions, and keep only the top-3 runs per banning region.
    """
    runs = get_banning_region_from_runs(runs)

    records = []

    for _, row in runs.iterrows():
        folder = row["folder_name"]
        region = row["region"]

        loss_path = Path(BASE_PATH) / folder / "loss_summary.csv"

        if loss_path.exists():
            try:
                df_loss = pd.read_csv(loss_path)
                households_loss_thousands = df_loss.loc[0, "households"]
            except Exception:
                households_loss_thousands = np.nan
        else:
            households_loss_thousands = np.nan

        households_loss_trillions = (
            households_loss_thousands * SCALE_TO_TRILLIONS
            if pd.notna(households_loss_thousands)
            else np.nan
        )

        records.append(
            {
                "folder_name": folder,
                "region": region,
                "households_loss_trillions": households_loss_trillions,
            }
        )

    df_losses = pd.DataFrame(records)

    df_valid = df_losses.dropna(subset=["households_loss_trillions"])
    if df_valid.empty:
        raise ValueError("No valid loss_summary data to select top runs.")

    df_sorted = df_valid.sort_values(
        by=["region", "households_loss_trillions"],
        ascending=[True, False]
    )

    top3 = df_sorted.groupby("region", group_keys=False).head(3)

    top3_runs = runs.merge(
        top3[["folder_name", "region", "households_loss_trillions"]],
        on=["folder_name", "region"],
        how="inner"
    )

    return top3_runs


def main():
    print("=" * 80)
    print("SELF-DAMAGE SHARE ANALYSIS: EXPORT-BAN REGION'S OWN DAMAGE SHARE IN TOP3 RUNS")
    print("=" * 80)

    # --- 1. Load Total Economic Output (optional here, but we keep for consistency) ---
    print("\n[1/4] Loading total region output data (not used in share metric but kept for consistency)...")
    try:
        df_output = pd.read_csv(TOTAL_OUTPUT_FILE)

        if df_output.columns[0] != "iso":
            df_output.rename(columns={df_output.columns[0]: "iso"}, inplace=True)
        if df_output.columns[1] != "total_output":
            df_output.rename(columns={df_output.columns[1]: "total_output"}, inplace=True)

        output_map = df_output.set_index("iso")["total_output"].to_dict()
        all_regions = sorted(df_output["iso"].unique())
        num_regions = len(all_regions)

        print(f"  Loaded output data for {num_regions} regions")
    except Exception as e:
        print(f"  ERROR loading {TOTAL_OUTPUT_FILE}: {e}")
        return

    # --- 2. Load Runs Mapping ---
    print("\n[2/4] Loading runs file...")
    try:
        runs = pd.read_csv(RUNS_FILE)

        if "exists_in_global4_final" in runs.columns:
            runs = runs[runs["exists_in_global4_final"] == True]

        if "folder_name" not in runs.columns:
            raise ValueError("RUNS_FILE must contain 'folder_name'.")

        runs = get_banning_region_from_runs(runs)

        print(f"  Found {len(runs)} valid simulations")
    except Exception as e:
        print(f"  ERROR loading {RUNS_FILE}: {e}")
        return

    # --- 3. Select top-3 runs per export-banning region ---
    print("\n[3/4] Selecting top-3 runs per export-banning region...")
    try:
        top3_runs = select_top3_runs_per_region(runs)
        print(f"  Selected {len(top3_runs)} runs (<= 3 per banning region)")
    except Exception as e:
        print(f"  ERROR selecting top-3 runs per region: {e}")
        return

    # --- 4. For each selected run, compute banning region's self-damage share ---
    print("\n[4/4] Computing self-damage share per selected run...")

    # To store per-run self-damage share
    per_run_records: List[Dict[str, object]] = []

    processed_count = 0

    for _, row in top3_runs.iterrows():
        folder = row["folder_name"]
        banning_region = row["region"]
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if not loss_path.exists():
            print(f"\n  WARNING: {folder} - loss_per_region_sector_time.csv not found")
            continue

        print(f"  [{processed_count+1:4d}/{len(top3_runs)}] {folder}", end="\r")

        try:
            df_loss = pd.read_csv(loss_path)

            if "region" not in df_loss.columns or "loss" not in df_loss.columns:
                del df_loss
                continue

            # Sum total losses in this simulation (all regions)
            total_loss = df_loss["loss"].sum()

            # Sum losses for the banning region itself (self-damage)
            self_loss = df_loss.loc[df_loss["region"] == banning_region, "loss"].sum()

            if total_loss > 0:
                self_share = self_loss / total_loss
            else:
                self_share = np.nan

            per_run_records.append(
                {
                    "folder_name": folder,
                    "banning_region": banning_region,
                    "total_loss": float(total_loss),
                    "self_loss": float(self_loss),
                    "self_share": float(self_share) if not np.isnan(self_share) else np.nan,
                    "households_loss_trillions": row.get("households_loss_trillions", np.nan),
                }
            )

            del df_loss
            gc.collect()
            processed_count += 1

        except Exception as e:
            print(f"\n  ERROR processing {folder}: {e}")

    print(f"\n  Completed processing {processed_count} selected simulation files")

    if not per_run_records:
        print("No valid per-run records computed. Exiting.")
        return

    df_per_run = pd.DataFrame(per_run_records)

    # Compute average self-damage share per banning region (across its top3 runs)
    df_region_avg = (
        df_per_run
        .groupby("banning_region", as_index=False)["self_share"]
        .mean()
        .rename(columns={"self_share": "avg_self_share_top3"})
    )

    # Compute global average self-damage share across all runs / regions
    global_avg_self_share = df_per_run["self_share"].mean(skipna=True)

    # Attach global average as a small separate row / table
    df_global = pd.DataFrame(
        {
            "metric": ["global_avg_self_share_top3_runs"],
            "value": [global_avg_self_share],
        }
    )

    # Save results: per-region averages + per-run details + global average
    try:
        # 1) Per-region averages
        region_avg_file = OUT_FILE.replace(".csv", "_per_region.csv")
        df_region_avg.to_csv(region_avg_file, index=False)

        # 2) Per-run details (for inspection)
        per_run_file = OUT_FILE.replace(".csv", "_per_run.csv")
        df_per_run.to_csv(per_run_file, index=False)

        # 3) Global average
        global_avg_file = OUT_FILE.replace(".csv", "_global_avg.csv")
        df_global.to_csv(global_avg_file, index=False)

        print("\n" + "=" * 80)
        print(f"DONE! Results saved to:")
        print(f"  Per-region averages: {region_avg_file}")
        print(f"  Per-run details   : {per_run_file}")
        print(f"  Global average    : {global_avg_file}")
        print("=" * 80)
        print(f"Global average self-damage share across all top3 runs: {global_avg_self_share:.6f}")

    except Exception as e:
        print(f"\nERROR saving output files: {e}")


if __name__ == "__main__":
    main()