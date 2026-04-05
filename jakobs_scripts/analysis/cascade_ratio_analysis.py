import os
import pandas as pd
import numpy as np


# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final8weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status2.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
MRIO_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
OUT_FILE = "loss_rankings_cascade_ratio.csv"


def compute_exports_for_region_sector(io_df, region_sector):
    """
    For a single (region, sector) pair, compute total exports to other regions.
    Returns a float value.
    """
    # Parse region_sector (format: "region_sector")
    region, sector = region_sector.split("_", 1)
    
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")
    
    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0
    
    # Find the row for this specific region-sector
    try:
        # Get the row index location
        row_loc = (io_df.index.get_level_values(row_region_level) == region) & \
                  (io_df.index.get_level_values(row_sector_level) == sector)
        
        if not row_loc.any():
            print(f"Warning: Region-sector {region_sector} not found in MRIO")
            return np.nan
        
        # Total output for this region-sector (sum across all destinations)
        row_total = io_df.loc[row_loc].sum().sum()
        
        # Domestic part: sum of flows to same region's columns
        col_mask = io_df.columns.get_level_values(col_region_level) == region
        domestic = io_df.loc[row_loc, col_mask].sum().sum()
        
        # Exports to other regions = total - domestic
        exports_other = row_total - domestic
        
        return exports_other
        
    except Exception as e:
        print(f"Error computing exports for {region_sector}: {e}")
        return np.nan


def main():
    # --- 1. Load mapping and sector table ---
    print("Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]

    print("Loading sector table...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)

    # --- 2. Build base region–sector loss table (one row per simulation) ---
    print("Processing loss summaries...")
    records = []

    for _, row in runs.iterrows():
        region_sector = row["region_sector"]
        folder = row["folder_name"]

        # Split only on first underscore
        region, sector = region_sector.split("_", 1)

        loss_path = os.path.join(BASE_PATH, folder, "loss_summary.csv")

        if os.path.exists(loss_path):
            df_loss = pd.read_csv(loss_path)
            # this is in thousands USD
            households_loss_thousands = df_loss.loc[0, "households"]
        else:
            households_loss_thousands = float("nan")

        records.append(
            {
                "region": region,
                "sector": sector,
                "region_sector": region_sector,
                "households_loss": households_loss_thousands,
            }
        )

    all_rs = pd.DataFrame(records)

    # Map to sector types
    sector_type_map = sector_table.set_index("sector")["type"]
    all_rs["sector_type"] = all_rs["sector"].map(sector_type_map)

    # --- 3. Load MRIO once and compute exports for each region-sector ---
    print("Loading MRIO file (3GB)...")
    io_df = pd.read_pickle(MRIO_FILE)
    print("Computing exports for each region-sector...")
    
    # Add column for exports
    all_rs["initial_exports_other_regions"] = np.nan
    
    # Process in chunks to show progress
    total_rows = len(all_rs)
    chunk_size = 100
    
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        
        for idx in range(i, end_idx):
            region_sector = all_rs.loc[idx, "region_sector"]
            exports = compute_exports_for_region_sector(io_df, region_sector)
            all_rs.loc[idx, "initial_exports_other_regions"] = exports
        
        print(f"  Processed {end_idx}/{total_rows} region-sectors")

    # --- 4. Compute cascade ratio ---
    print("Computing cascade ratios...")
    # Handle zero or missing exports to avoid divide-by-zero
    zero_or_missing = (
        all_rs["initial_exports_other_regions"].isna()
        | (all_rs["initial_exports_other_regions"] == 0)
    )

    all_rs["cascade_ratio"] = float("nan")
    all_rs.loc[~zero_or_missing, "cascade_ratio"] = (
        all_rs.loc[~zero_or_missing, "households_loss"]
        / all_rs.loc[~zero_or_missing, "initial_exports_other_regions"]
    )

    # --- 5. Rankings based on cascade ratio ---
    print("Generating rankings...")
    
    # 1) Region–sector ranking (per simulation)
    rs_rank = (
        all_rs[["region_sector", "cascade_ratio"]]
        .sort_values("cascade_ratio", ascending=False)
        .reset_index(drop=True)
    )
    rs_rank["region_sector_n_runs"] = 1  # one simulation per row
    rs_rank.columns = [
        "region_sector",
        "region_sector_cascade_ratio",
        "region_sector_n_runs",
    ]

    # 2) Region ranking (average cascade ratio per region)
    by_region = (
        all_rs.groupby("region", as_index=False)
        .agg(
            total_cascade_ratio=("cascade_ratio", "sum"),
            n_runs=("cascade_ratio", "size"),
        )
    )
    by_region["region_avg_cascade_ratio"] = (
        by_region["total_cascade_ratio"] / by_region["n_runs"]
    )
    by_region = (
        by_region.sort_values("region_avg_cascade_ratio", ascending=False)
        .reset_index(drop=True)
    )
    by_region = by_region[["region", "region_avg_cascade_ratio", "n_runs"]]
    by_region.columns = ["region", "region_avg_cascade_ratio", "region_n_runs"]

    # 3) Sector ranking (average cascade ratio per sector)
    by_sector = (
        all_rs.groupby("sector", as_index=False)
        .agg(
            total_cascade_ratio=("cascade_ratio", "sum"),
            n_runs=("cascade_ratio", "size"),
        )
    )
    by_sector["sector_avg_cascade_ratio"] = (
        by_sector["total_cascade_ratio"] / by_sector["n_runs"]
    )
    by_sector = (
        by_sector.sort_values("sector_avg_cascade_ratio", ascending=False)
        .reset_index(drop=True)
    )
    by_sector = by_sector[["sector", "sector_avg_cascade_ratio", "n_runs"]]
    by_sector.columns = ["sector", "sector_avg_cascade_ratio", "sector_n_runs"]

    # 4) Sector-type ranking (average cascade ratio per sector type)
    by_sector_type = (
        all_rs.groupby("sector_type", as_index=False)
        .agg(
            total_cascade_ratio=("cascade_ratio", "sum"),
            n_runs=("cascade_ratio", "size"),
        )
    )
    by_sector_type["sector_type_avg_cascade_ratio"] = (
        by_sector_type["total_cascade_ratio"] / by_sector_type["n_runs"]
    )
    by_sector_type = (
        by_sector_type.sort_values("sector_type_avg_cascade_ratio", ascending=False)
        .reset_index(drop=True)
    )
    by_sector_type = by_sector_type[
        ["sector_type", "sector_type_avg_cascade_ratio", "n_runs"]
    ]
    by_sector_type.columns = [
        "sector_type",
        "sector_type_avg_cascade_ratio",
        "sector_type_n_runs",
    ]

    # --- 6. Pad to same length and export in wide layout ---
    print("Writing output file...")
    
    max_len = max(len(rs_rank), len(by_region), len(by_sector), len(by_sector_type))
    rs_rank = rs_rank.reindex(range(max_len))
    by_region = by_region.reindex(range(max_len))
    by_sector = by_sector.reindex(range(max_len))
    by_sector_type = by_sector_type.reindex(range(max_len))

    final_df = pd.DataFrame(
        {
            "region_sector": rs_rank["region_sector"],
            "region_sector_cascade_ratio": rs_rank["region_sector_cascade_ratio"],
            "region_sector_n_runs": rs_rank["region_sector_n_runs"],
            "sep1": pd.NA,
            "region": by_region["region"],
            "region_avg_cascade_ratio": by_region["region_avg_cascade_ratio"],
            "region_n_runs": by_region["region_n_runs"],
            "sep2": pd.NA,
            "sector": by_sector["sector"],
            "sector_avg_cascade_ratio": by_sector["sector_avg_cascade_ratio"],
            "sector_n_runs": by_sector["sector_n_runs"],
            "sep3": pd.NA,
            "sector_type": by_sector_type["sector_type"],
            "sector_type_avg_cascade_ratio": by_sector_type[
                "sector_type_avg_cascade_ratio"
            ],
            "sector_type_n_runs": by_sector_type["sector_type_n_runs"],
        }
    )

    final_df.to_csv(OUT_FILE, index=False)
    print(f"Done! Results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()