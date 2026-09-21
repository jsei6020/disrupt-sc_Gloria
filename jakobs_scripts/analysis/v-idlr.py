import gc
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
BASE_PATH = Path("/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks")
RUNS_FILE = Path("/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_threat.csv")
REGIONS_TABLE_FILE = Path("/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv")

# Input file for Total Output (as provided in context)
TOTAL_OUTPUT_FILE = Path("/home/user/Documents/University/Master Thesis/vulnerability_analysis/total_region_output.csv")

# Output files
OUT_FILE_BASIC = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-idlr_threat.csv"
OUT_FILE_EXTENDED = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/extended_vulnerability_analysis_threat.csv"

TIME_COL_CANDIDATES = ["time_step"]
MAX_DIRECT_TIMESTEP = 12


def detect_time_column(df: pd.DataFrame):
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: TOTAL INDIRECT/DIRECT LOSS RATIO BY REGION")
    print("=" * 80)

    # --- 1. Load Runs Mapping ---
    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    runs = runs[
        (runs["exists_in_global4_final"] == True) &
        (runs["region_sector"] != "UNKNOWN (extra folder)")
    ].copy()
    print(f"  Found {len(runs)} valid simulations")

    required_run_cols = {"folder_name", "region_sector"}
    if not required_run_cols.issubset(runs.columns):
        raise ValueError("Expected 'folder_name' and 'region_sector' columns in runs file")

    runs["folder_name"] = runs["folder_name"].astype(str)
    runs["region_sector"] = runs["region_sector"].astype(str)

    # --- 2. Load Region Name Mapping ---
    print("\n[2/5] Loading region table...")
    regions_df = pd.read_csv(REGIONS_TABLE_FILE)

    required_region_cols = {"Region_acronyms", "Region_names"}
    if not required_region_cols.issubset(regions_df.columns):
        raise ValueError("Expected 'Region_acronyms' and 'Region_names' columns in regions_table.csv")

    region_name_map = dict(
        zip(
            regions_df["Region_acronyms"].astype(str),
            regions_df["Region_names"].astype(str)
        )
    )
    print(f"  Loaded region names for {len(region_name_map)} regions")

    # --- 3. Process loss files and collect per-run region totals ---
    print("\n[3/5] Processing loss files...")

    all_region_losses = []
    processed_count = 0
    skipped_count = 0

    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        disrupted_region_sector = row["region_sector"]
        loss_path = BASE_PATH / folder / "loss_per_region_sector_time.csv"

        if not loss_path.exists():
            if skipped_count < 5 or idx == len(runs) - 1:
                print(f"  WARNING: {folder} - file not found")
            skipped_count += 1
            continue

        if idx % 50 == 0:
            print(f"  Processing run {idx+1}/{len(runs)}...")

        try:
            df_loss = pd.read_csv(loss_path)

            required_loss_cols = {"region", "loss"}
            if not required_loss_cols.issubset(df_loss.columns):
                print(f"\n  WARNING: {folder} missing required columns {required_loss_cols}")
                skipped_count += 1
                del df_loss
                gc.collect()
                continue

            time_col = detect_time_column(df_loss)
            if time_col is None:
                print(f"\n  WARNING: {folder} has no time column among {TIME_COL_CANDIDATES}")
                skipped_count += 1
                del df_loss
                gc.collect()
                continue

            df_loss["region"] = df_loss["region"].astype(str)
            df_loss["loss"] = pd.to_numeric(df_loss["loss"], errors="coerce").fillna(0.0)
            df_loss[time_col] = pd.to_numeric(df_loss[time_col], errors="coerce")
            
            # Ensure sector column exists for masking
            if "sector" not in df_loss.columns:
                 # Fallback if sector is missing, though usually present in this schema
                 print(f"\n  WARNING: {folder} missing 'sector' column")
                 skipped_count += 1
                 del df_loss
                 gc.collect()
                 continue

            df_loss["region_sector"] = df_loss["sector"].astype(str)

            # Direct loss:
            # same region_sector as disrupted one AND within first 12 time steps
            direct_mask = (
                (df_loss["region_sector"] == disrupted_region_sector) &
                (df_loss[time_col] <= MAX_DIRECT_TIMESTEP)
            )
            
            df_loss["direct_loss_component"] = np.where(
                direct_mask,
                df_loss["loss"],
                0.0
            )

            df_loss["indirect_loss_component"] = np.where(
                ~direct_mask,
                df_loss["loss"],
                0.0
            )

            grouped = (
                df_loss.groupby("region", as_index=False)
                .agg(
                    total_direct_loss=("direct_loss_component", "sum"),
                    total_indirect_loss=("indirect_loss_component", "sum")
                )
            )
            grouped["run_folder"] = folder
            all_region_losses.append(
                grouped[["run_folder", "region", "total_direct_loss", "total_indirect_loss"]]
            )

            processed_count += 1

            del df_loss
            del grouped
            gc.collect()

        except Exception as e:
            print(f"\n  ERROR processing {folder}: {e}")
            skipped_count += 1

    print(f"\n  Completed processing {processed_count} simulation files")
    print(f"  Skipped {skipped_count} simulation files")

    if not all_region_losses:
        print("\nERROR: No losses could be aggregated.")
        return

    # --- 4. Aggregate totals by affected region and compute ratio ---
    print("\n[4/5] Aggregating total losses by region...")

    df_all = pd.concat(all_region_losses, ignore_index=True)

    df_results = (
        df_all.groupby("region", as_index=False)
        .agg(
            total_direct_loss=("total_direct_loss", "sum"),
            total_indirect_loss=("total_indirect_loss", "sum"),
            run_count=("run_folder", "nunique")
        )
        .rename(columns={"region": "iso"})
    )

    df_results["indirect_direct_ratio"] = np.where(
        df_results["total_direct_loss"] > 0,
        df_results["total_indirect_loss"] / df_results["total_direct_loss"],
        np.nan
    )

    df_results["region_name"] = df_results["iso"].map(region_name_map)

    df_results = df_results[
        [
            "iso",
            "region_name",
            "total_direct_loss",
            "total_indirect_loss",
            "indirect_direct_ratio",
            "run_count",
        ]
    ]

    df_results = df_results.sort_values(
        by="indirect_direct_ratio",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    # --- 4b. Extended Analysis: Split Output Ratios ---
    print("\n[4b/5] Calculating extended vulnerability metrics (Loss/Output)...")
    
    try:
        # Load Total Output
        if not TOTAL_OUTPUT_FILE.exists():
            raise FileNotFoundError(f"Total output file not found at {TOTAL_OUTPUT_FILE}")
        
        df_output = pd.read_csv(TOTAL_OUTPUT_FILE)
        
        # Ensure ISO consistency
        df_output["iso"] = df_output["iso"].astype(str)
        
        # Merge vulnerability results with total output
        df_extended = pd.merge(df_results, df_output, on="iso", how="inner")
        
        # Calculate Ratios
        # Avoid division by zero
        df_extended["indirect_loss_to_output_ratio"] = np.where(
            df_extended["total_output"] > 0,
            df_extended["total_indirect_loss"] / df_extended["total_output"],
            np.nan
        )
        
        df_extended["direct_loss_to_output_ratio"] = np.where(
            df_extended["total_output"] > 0,
            df_extended["total_direct_loss"] / df_extended["total_output"],
            np.nan
        )
        
        # Create Table 1: Sorted by Indirect Loss / Total Output (Descending)
        df_indirect_sorted = df_extended[["iso", "region_name", "indirect_loss_to_output_ratio"]].copy()
        df_indirect_sorted = df_indirect_sorted.sort_values(by="indirect_loss_to_output_ratio", ascending=False)
        df_indirect_sorted = df_indirect_sorted.reset_index(drop=True)
        
        # Create Table 2: Sorted by Direct Loss / Total Output (Descending)
        df_direct_sorted = df_extended[["iso", "region_name", "direct_loss_to_output_ratio"]].copy()
        df_direct_sorted = df_direct_sorted.sort_values(by="direct_loss_to_output_ratio", ascending=False)
        df_direct_sorted = df_direct_sorted.reset_index(drop=True)
        
        # Add Separator Columns
        # We create a column of empty strings or NaNs to act as a visual separator
        separator_col_1 = pd.Series([""] * len(df_indirect_sorted), name="---")
        separator_col_2 = pd.Series([""] * len(df_direct_sorted), name="---")
        
        # Insert separators
        df_indirect_final = df_indirect_sorted.copy()
        df_indirect_final.insert(0, "---", separator_col_1.values)
        
        df_direct_final = df_direct_sorted.copy()
        df_direct_final.insert(0, "---", separator_col_2.values)
        
        # Combine side-by-side
        # Note: Both dataframes have the same length because they originate from the same merged set
        final_table = pd.concat([df_indirect_final, df_direct_final], axis=1)
        
        # Rename columns for clarity in the final CSV if desired, 
        # but keeping original request structure:
        # --- | iso | region_name | indirect_ratio | --- | iso | region_name | direct_ratio
        
        print("  Extended table created successfully.")
        
    except Exception as e:
        print(f"  ERROR in extended analysis: {e}")
        final_table = None

    # --- 5. Save output ---
    print("\n[5/5] Saving results...")
    
    # Save basic results
    df_results.to_csv(OUT_FILE_BASIC, index=False)
    print(f"  Basic results saved to {OUT_FILE_BASIC}")
    
    # Save extended results if successful
    if final_table is not None:
        final_table.to_csv(OUT_FILE_EXTENDED, index=False)
        print(f"  Extended results saved to {OUT_FILE_EXTENDED}")

    print(f"\n{'=' * 80}")
    print(f"DONE!")
    print(f"{'=' * 80}")

    print("\nTOP 10 REGIONS BY TOTAL INDIRECT/DIRECT LOSS RATIO:")
    print(
        df_results[
            [
                "iso",
                "region_name",
                "indirect_direct_ratio",
                "total_direct_loss",
                "total_indirect_loss",
                "run_count",
            ]
        ]
        .head(10)
        .to_string(index=True)
    )
    
    if final_table is not None:
        print("\nTOP 5 REGIONS BY INDIRECT LOSS / TOTAL OUTPUT:")
        print(final_table.iloc[:5, 1:4].to_string(index=False)) # Skip first separator col
        print("\nTOP 5 REGIONS BY DIRECT LOSS / TOTAL OUTPUT:")
        print(final_table.iloc[:5, 5:8].to_string(index=False)) # Skip first half + separator


if __name__ == "__main__":
    main()