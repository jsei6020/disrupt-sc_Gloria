import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
import re

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_threat.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"
OUTPUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/total_region_output.csv"
OUT_FILE_CSV = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-vs_rt_threat.csv"

HORIZON = 26


def compute_region_time_metrics(df_loss: pd.DataFrame) -> pd.DataFrame:
    """
    For one simulation:
      - aggregate loss per region & time_step,
      - compute cumulative loss per region,
      - find first time step with non-zero loss (t_start),
      - find time step where cumulative loss >= 90% of total (t_90),
      - compute:
          * time_to_90pct_region = t_90 - t_start
          * mult_metric = total_loss_region * time_to_90pct_region
          * div_metric  = total_loss_region / (HORIZON - time_to_90pct_region)
    """
    if df_loss.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "total_loss_region",
                "time_to_90pct_region",
                "mult_metric_region",
                "div_metric_region",
            ]
        )

    # 1. Aggregate loss per region & time_step
    loss_rt = (
        df_loss.groupby(["region", "time_step"], as_index=False)["loss"]
        .sum()
    )

    # 2. Sort to ensure cumulative sum and time ordering are correct
    loss_rt = loss_rt.sort_values(["region", "time_step"])

    # 3. Compute cumulative loss per region
    loss_rt["cumulative_loss"] = (
        loss_rt.groupby("region")["loss"].cumsum()
    )

    # 4. Get total loss per region
    total_by_region = (
        loss_rt.groupby("region", as_index=False)["cumulative_loss"]
        .max()
        .rename(columns={"cumulative_loss": "total_loss_region"})
    )

    # 5. Merge total loss back to main df
    loss_rt = loss_rt.merge(total_by_region, on="region", how="left")

    # 6. Calculate 90% threshold
    loss_rt["threshold_90"] = 0.9 * loss_rt["total_loss_region"]

    # A. Find t_start: first time_step where loss > 0 for each region
    df_nonzero = loss_rt[loss_rt["loss"] > 0].copy()
    if df_nonzero.empty:
        t_start_df = pd.DataFrame(columns=["region", "t_start"])
    else:
        df_nonzero = df_nonzero.sort_values(["region", "time_step"])
        t_start_df = df_nonzero.drop_duplicates(
            subset=["region"], keep="first"
        )[["region", "time_step"]]
        t_start_df.rename(columns={"time_step": "t_start"}, inplace=True)

    # B. Find t_90: first time_step where cumulative_loss >= 90% of total
    df_meets = loss_rt[loss_rt["cumulative_loss"] >= loss_rt["threshold_90"]].copy()
    if df_meets.empty:
        t_90_df = pd.DataFrame(columns=["region", "t_90"])
    else:
        df_meets = df_meets.sort_values(["region", "time_step"])
        t_90_df = df_meets.drop_duplicates(
            subset=["region"], keep="first"
        )[["region", "time_step"]]
        t_90_df.rename(columns={"time_step": "t_90"}, inplace=True)

    # C. Combine t_start and t_90
    region_times = total_by_region[["region"]].copy()
    region_times = region_times.merge(t_start_df, on="region", how="left")
    region_times = region_times.merge(t_90_df, on="region", how="left")

    # D. Calculate time_to_90
    region_times["time_to_90pct_region"] = np.where(
        region_times["t_start"].notna() & region_times["t_90"].notna(),
        region_times["t_90"] - region_times["t_start"],
        np.nan,
    )
    region_times.drop(columns=["t_start", "t_90"], inplace=True)

    # E. Merge calculated times back to total_by_region
    final_metrics = total_by_region.merge(
        region_times[["region", "time_to_90pct_region"]],
        on="region",
        how="left",
    )

    # 9. Calculate metrics
    final_metrics["mult_metric_region"] = (
        final_metrics["total_loss_region"] * final_metrics["time_to_90pct_region"]
    )

    denominator = HORIZON - final_metrics["time_to_90pct_region"]
    valid_mask = (
        final_metrics["time_to_90pct_region"].notna() & (denominator > 0)
    )

    final_metrics["div_metric_region"] = np.where(
        valid_mask,
        final_metrics["total_loss_region"] / denominator,
        np.nan,
    )

    return final_metrics[
        [
            "region",
            "total_loss_region",
            "time_to_90pct_region",
            "mult_metric_region",
            "div_metric_region",
        ]
    ]


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: ZERO-FILLED & STANDARDIZED METRICS")
    print("=" * 80)

    # --- 1. Load runs file and FILTER out UNKNOWN entries ---
    print("\n[1/6] Loading and filtering runs file...")
    runs = pd.read_csv(RUNS_FILE)

    initial_count = len(runs)

    # Filter: exclude rows where 'region_sector' starts with "UNKNOWN" (case-insensitive)
    if "region_sector" in runs.columns:
        mask_valid = ~runs["region_sector"].astype(str).str.upper().str.startswith(
            "UNKNOWN"
        )
        runs = runs[mask_valid]
        filtered_out = initial_count - len(runs)
        if filtered_out > 0:
            print(f"  Filtered out {filtered_out} entries starting with 'UNKNOWN'")
    else:
        print("  WARNING: 'region_sector' column not found. Skipping UNKNOWN filter.")

    # Additional check for exists_in_global4_final if it exists
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]

    print(f"  Found {len(runs)} valid simulations to process")

    if runs.empty:
        print("ERROR: No valid simulations remaining after filtering.")
        return

    # --- 2. Load region names ---
    print("\n[2/6] Loading region table...")
    try:
        regions_df = pd.read_csv(REGIONS_TABLE_FILE)
        if (
            "Region_acronyms" not in regions_df.columns
            or "Region_names" not in regions_df.columns
        ):
            raise ValueError(
                "Expected 'Region_acronyms' and 'Region_names' columns in regions_table.csv"
            )

        region_name_map = dict(
            zip(
                regions_df["Region_acronyms"].astype(str).str.strip(),
                regions_df["Region_names"].astype(str).str.strip(),
            )
        )
        # Get list of ALL expected regions
        all_regions_list = (
            regions_df["Region_acronyms"].astype(str).str.strip().unique()
        )
        print(f"  Loaded region names for {len(region_name_map)} regions")
    except Exception as e:
        print(f"  ERROR loading {REGIONS_TABLE_FILE}: {e}")
        return

    # --- 3. Load economic output data ---
    print("\n[3/6] Loading total region output data...")
    try:
        df_output = pd.read_csv(OUTPUT_FILE)

        if df_output.columns[0] != "iso":
            df_output.rename(columns={df_output.columns[0]: "iso"}, inplace=True)
        if df_output.columns[1] != "total_output":
            df_output.rename(columns={df_output.columns[1]: "total_output"}, inplace=True)

        df_output["iso"] = df_output["iso"].astype(str).str.strip()
        df_output["total_output"] = pd.to_numeric(
            df_output["total_output"], errors="coerce"
        )

        output_map = df_output.set_index("iso")["total_output"].to_dict()
        print(f"  Loaded economic data for {len(output_map)} regions")
    except Exception as e:
        print(f"  ERROR loading {OUTPUT_FILE}: {e}")
        return

    all_region_metrics = []
    processed_count = 0

    # --- 4. Process each simulation ---
    print("\n[4/6] Processing loss files (zero-filling missing regions)...")

    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")

            try:
                df_loss = pd.read_csv(loss_path)

                required_cols = {"region", "time_step", "loss"}
                if not required_cols.issubset(df_loss.columns):
                    print(f"\n  WARNING: {folder} missing required columns, skipping")
                    del df_loss
                    gc.collect()
                    continue

                # 1. Compute metrics for regions PRESENT in this run
                region_metrics = compute_region_time_metrics(df_loss)

                # 2. Identify MISSING regions for this run
                present_regions = set(
                    region_metrics["region"].astype(str).str.strip()
                )
                missing_regions = set(all_regions_list) - present_regions

                # 3. Create zero-filled rows for missing regions
                if missing_regions:
                    missing_df = pd.DataFrame(
                        {
                            "region": list(missing_regions),
                            "total_loss_region": 0.0,
                            "time_to_90pct_region": 0.0,
                            # As requested: 0 if not mentioned
                            "mult_metric_region": 0.0,
                            "div_metric_region": 0.0,
                        }
                    )
                    # Concatenate present and missing
                    region_metrics = pd.concat(
                        [region_metrics, missing_df], ignore_index=True
                    )

                # Ensure region column is string and stripped for consistency
                region_metrics["region"] = (
                    region_metrics["region"].astype(str).str.strip()
                )
                region_metrics["run_folder"] = folder

                all_region_metrics.append(region_metrics)

                del df_loss
                del region_metrics
                gc.collect()
                processed_count += 1

            except Exception as e:
                print(f"\n  ERROR processing {folder}: {e}")
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")

    print(f"\n  Completed processing {processed_count} simulation files")

    if not all_region_metrics:
        print("\nERROR: No region-level metrics calculated.")
        return

    # --- 5. Aggregate across runs ---
    print("\n[5/6] Aggregating metrics...")

    df_all = pd.concat(all_region_metrics, ignore_index=True)

    df_avg = (
        df_all.groupby("region", as_index=False)
        .agg(
            avg_total_loss_region=("total_loss_region", "mean"),
            avg_mult_metric=("mult_metric_region", "mean"),
            avg_div_metric=("div_metric_region", "mean"),
            avg_time_to_90pct=("time_to_90pct_region", "mean"),
            n_runs=("run_folder", "nunique"),
        )
    )

    df_avg["region"] = df_avg["region"].astype(str).str.strip()

    # Map Region Names
    df_avg["region_name"] = df_avg["region"].map(region_name_map)

    # Map Total Output for Standardization
    df_avg["total_output"] = df_avg["region"].map(output_map)

    # --- STANDARDIZATION LOGIC ---
    # Standardize total loss, mult, and div metrics by total_output
    df_avg["avg_total_loss_region_std"] = np.where(
        df_avg["total_output"].notna() & (df_avg["total_output"] > 0),
        df_avg["avg_total_loss_region"] / df_avg["total_output"],
        np.nan,
    )

    df_avg["avg_mult_metric_std"] = np.where(
        df_avg["total_output"].notna() & (df_avg["total_output"] > 0),
        df_avg["avg_mult_metric"] / df_avg["total_output"],
        np.nan,
    )

    df_avg["avg_div_metric_std"] = np.where(
        df_avg["total_output"].notna() & (df_avg["total_output"] > 0),
        df_avg["avg_div_metric"] / df_avg["total_output"],
        np.nan,
    )

    # --- 6. Create Side-by-Side CSV ---
    print("\n[6/6] Creating side-by-side ranking CSV...")

    # 0. Sort by standardized total loss (Intensity-style metric)
    rank_loss = df_avg.sort_values(
        by="avg_total_loss_region_std",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # 1. Sort by Standardized Multiplication Metric (Descending)
    rank_mult = df_avg.sort_values(
        by="avg_mult_metric_std",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # 2. Sort by Standardized Division Metric (Descending)
    rank_div = df_avg.sort_values(
        by="avg_div_metric_std",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # 3. Sort by Time to 90% (Descending)
    rank_t90 = df_avg.sort_values(
        by="avg_time_to_90pct",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # Define base columns for each block
    base_cols = ["region", "region_name"]

    # Block 0: standardized average total loss
    df_block0 = rank_loss[base_cols + ["avg_total_loss_region_std"]].copy()
    df_block0.columns = ["iso_0", "region_name_0", "metric_loss_std"]

    # Block 1: Mult Metric
    df_block1 = rank_mult[base_cols + ["avg_mult_metric_std"]].copy()
    df_block1.columns = ["iso_1", "region_name_1", "metric_mult_std"]

    # Block 2: Div Metric
    df_block2 = rank_div[base_cols + ["avg_div_metric_std"]].copy()
    df_block2.columns = ["iso_2", "region_name_2", "metric_div_std"]

    # Block 3: Time to 90%
    df_block3 = rank_t90[base_cols + ["avg_time_to_90pct"]].copy()
    df_block3.columns = ["iso_3", "region_name_3", "metric_time_90"]

    # Determine max length to align rows
    max_len = max(len(df_block0), len(df_block1), len(df_block2), len(df_block3))

    # Reindex to ensure all blocks have same number of rows
    df_block0 = df_block0.reindex(range(max_len))
    df_block1 = df_block1.reindex(range(max_len))
    df_block2 = df_block2.reindex(range(max_len))
    df_block3 = df_block3.reindex(range(max_len))

    # Empty separator column
    separator_col = pd.Series([""] * max_len, name="---")

    # Concatenate horizontally
    df_final = pd.concat(
        [
            df_block0,
            separator_col,
            df_block1,
            separator_col,
            df_block2,
            separator_col,
            df_block3,
        ],
        axis=1,
    )

    # Save to CSV
    df_final.to_csv(OUT_FILE_CSV, index=False)
    print(f"  Saved side-by-side CSV to: {OUT_FILE_CSV}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)

    # Print Top 5 for quick check
    print("\n📊 TOP 5 (Std. Avg Total Loss):")
    print(
        rank_loss[["region", "region_name", "avg_total_loss_region_std"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n📊 TOP 5 (Standardized Mult Metric):")
    print(
        rank_mult[["region", "region_name", "avg_mult_metric_std"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n📊 TOP 5 (Standardized Div Metric):")
    print(
        rank_div[["region", "region_name", "avg_div_metric_std"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n📊 TOP 5 (Time to 90%):")
    print(
        rank_t90[["region", "region_name", "avg_time_to_90pct"]]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()