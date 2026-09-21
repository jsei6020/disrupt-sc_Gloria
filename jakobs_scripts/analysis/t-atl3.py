#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---

BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_all.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"

OUT_FILE = "/home/user/Documents/University/Master Thesis/threat_analysis/t-atl3.csv"

# thousands USD -> trillions USD
SCALE_TO_TRILLIONS = 1e-9


def get_topN_avg_per_group(df, group_col, value_col, N):
    """
    For each group in group_col, take the top N rows by value_col (descending),
    then return the mean of those top N values per group.
    If a group has fewer than N observations, use all available.
    """
    df_valid = df.dropna(subset=[group_col, value_col])
    if df_valid.empty:
        return pd.DataFrame()

    df_sorted = df_valid.sort_values(by=[group_col, value_col],
                                     ascending=[True, False])
    topN = df_sorted.groupby(group_col, group_keys=False).head(N)

    avg_vals = topN.groupby(group_col)[value_col].mean().reset_index()
    return avg_vals


def get_top3_avg_ranking(df, group_col, value_col, label_prefix):
    """
    1. Group by group_col.
    2. Within each group, take top 3 rows by value_col (descending).
    3. Calculate mean of those top 3.
    4. Sort groups by this mean (descending).
    """
    df_valid = df.dropna(subset=[group_col, value_col])
    if df_valid.empty:
        return pd.DataFrame()

    df_sorted = df_valid.sort_values(by=[group_col, value_col],
                                     ascending=[True, False])
    top3_per_group = df_sorted.groupby(group_col, group_keys=False).head(3)

    avg_vals = top3_per_group.groupby(group_col)[value_col].mean().reset_index()
    avg_vals.columns = [group_col, f"{label_prefix}_AvgTop3Loss"]

    avg_vals = avg_vals.sort_values(
        by=f"{label_prefix}_AvgTop3Loss",
        ascending=False
    ).reset_index(drop=True)

    return avg_vals


def pad_df(df, max_len):
    """Pad a DataFrame with NaN rows to reach max_len rows."""
    if len(df) < max_len:
        padding = pd.DataFrame(
            np.nan,
            index=np.arange(max_len - len(df)),
            columns=df.columns
        )
        return pd.concat([df, padding], ignore_index=True)
    return df


def main():
    print("=" * 80)
    print("LOSS RANKINGS: REGION-SECTOR + TOP-3 AVG (REGION / SECTOR) + SECTOR-TYPE AVG (TOP100 / ALL)")
    print("=" * 80)

    # --- 1. Load runs, sector types, region name mapping ---

    print("\n[1/4] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)

    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]

    if "region_sector" not in runs.columns or "folder_name" not in runs.columns:
        raise ValueError("RUNS_FILE must have 'region_sector' and 'folder_name' columns.")

    runs["region"] = runs["region_sector"].str.split("_", n=1).str[0]
    runs["sector"] = runs["region_sector"].str.split("_", n=1).str[1]

    print(f"  Found {len(runs)} valid simulations")

    print("\nLoading sector table for sector_type mapping...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)
    if "sector" not in sector_table.columns or "type" not in sector_table.columns:
        raise ValueError("SECTOR_TABLE_FILE must contain 'sector' and 'type' columns.")

    sector_type_map = sector_table.set_index("sector")["type"].to_dict()
    runs["sector_type"] = runs["sector"].map(sector_type_map).fillna("Unknown")

    print("Loading regions table for region name mapping...")
    regions_df = pd.read_csv(REGIONS_TABLE_FILE)
    if "Region_acronyms" in regions_df.columns and "Region_names" in regions_df.columns:
        iso_col = "Region_acronyms"
        name_col = "Region_names"
    elif "iso" in regions_df.columns and "region_name" in regions_df.columns:
        iso_col = "iso"
        name_col = "region_name"
    else:
        raise ValueError(
            "REGIONS_TABLE_FILE must contain either "
            "('Region_acronyms','Region_names') or ('iso','region_name')."
        )

    region_name_map = dict(
        zip(regions_df[iso_col].astype(str), regions_df[name_col].astype(str))
    )

    # --- 2. Loop over runs, read loss_summary.csv, compute loss in trillions ---

    print("\n[2/4] Reading loss_summary.csv for each run...")
    records = []
    missing_count = 0

    for idx, row in runs.iterrows():
        region_sector = row["region_sector"]
        region = row["region"]
        sector = row["sector"]
        sector_type = row["sector_type"]
        folder = row["folder_name"]

        loss_path = Path(BASE_PATH) / folder / "loss_summary.csv"

        if loss_path.exists():
            try:
                df_loss = pd.read_csv(loss_path)
                households_loss_thousands = df_loss.loc[0, "households"]
            except Exception as e:
                print(f"  WARNING: Error reading {loss_path}: {e}")
                households_loss_thousands = np.nan
        else:
            households_loss_thousands = np.nan
            missing_count += 1

        households_loss_trillions = (
            households_loss_thousands * SCALE_TO_TRILLIONS
            if pd.notna(households_loss_thousands)
            else np.nan
        )

        records.append(
            {
                "region_sector": region_sector,
                "region": region,
                "sector": sector,
                "sector_type": sector_type,
                "households_loss_trillions": households_loss_trillions,
            }
        )

    print(f"  Total runs processed: {len(records)}")
    
    # --- AFTER THE LOOP ---

    # 1. Convert list of dicts to DataFrame
    df_results = pd.DataFrame(records)

    # 2. Calculate statistics
    col = "households_loss_trillions"

    average_loss = df_results[col].mean()
    median_loss = df_results[col].median()
    max_loss = df_results[col].max()
    min_loss = df_results[col].min()

    # 3. Print results
    print(f"  Average households loss (trillions): {average_loss}")
    print(f"  Median households loss (trillions):  {median_loss}")
    print(f"  Max households loss (trillions):     {max_loss}")
    print(f"  Min households loss (trillions):     {min_loss}")

    if missing_count > 0:
        print(f"  WARNING: {missing_count} loss_summary.csv files missing.")

    all_rs = pd.DataFrame(records)
    all_rs["region_name"] = (
        all_rs["region"].astype(str).map(region_name_map).fillna(all_rs["region"])
    )

    loss_col = "households_loss_trillions"

    # --- 3. Compute rankings / averages ---

    print("\n[3/4] Computing rankings...")

    # 3A. Region-sector ranking: only region_sector + metric
    rs_df = all_rs[["region_sector", loss_col]].copy()
    rs_df = rs_df.dropna(subset=[loss_col])
    rs_df = rs_df.sort_values(by=loss_col, ascending=False).reset_index(drop=True)

    regionsector_output = rs_df.rename(columns={
        loss_col: "region_sector_loss_trillions"
    })

    # 3B. Region top-3 average ranking (with region_name)
    region_top3 = get_top3_avg_ranking(
        all_rs, group_col="region", value_col=loss_col, label_prefix="Region"
    )
    region_top3["region_name"] = (
        region_top3["region"].astype(str).map(region_name_map).fillna(region_top3["region"])
    )

    region_output = region_top3[[
        "region",
        "region_name",
        "Region_AvgTop3Loss",
    ]].rename(columns={
        "region": "region_iso",
        "Region_AvgTop3Loss": "region_avg_top3_loss_trillions"
    })

    # 3C. Sector top-3 average ranking
    sector_top3 = get_top3_avg_ranking(
        all_rs, group_col="sector", value_col=loss_col, label_prefix="Sector"
    )

    sector_output = sector_top3[[
        "sector",
        "Sector_AvgTop3Loss",
    ]].rename(columns={
        "Sector_AvgTop3Loss": "sector_avg_top3_loss_trillions"
    })

    # 3D. Sector-type averages:
    #  - top 100 per sector_type
    #  - all observations per sector_type
    #  - number of observations (n_runs) per sector_type
    st_valid = all_rs.dropna(subset=["sector_type", loss_col])

    # Top-100 per sector_type
    st_top100 = get_topN_avg_per_group(
        st_valid, group_col="sector_type", value_col=loss_col, N=100
    )
    st_top100 = st_top100.rename(columns={
        "sector_type": "sector_type",
        "households_loss_trillions": "sector_type_avg_top100_loss_trillions"
    })

    # All observations per sector_type
    st_all = st_valid.groupby("sector_type")[loss_col].mean().reset_index()
    st_all = st_all.rename(columns={
        "households_loss_trillions": "sector_type_avg_all_loss_trillions"
    })

    # Count of observations per sector_type
    st_counts = st_valid.groupby("sector_type")[loss_col].count().reset_index()
    st_counts = st_counts.rename(columns={
        "households_loss_trillions": "sector_type_n_runs"
    })

    # Merge metrics
    sectortype_output = st_top100.merge(
        st_all, on="sector_type", how="outer"
    ).merge(
        st_counts, on="sector_type", how="outer"
    )

    # Sort sector types by top100 average (desc), then by all-average if needed
    sectortype_output = sectortype_output.sort_values(
        by=["sector_type_avg_top100_loss_trillions", "sector_type_avg_all_loss_trillions"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # --- 4. Combine into one wide file with separator columns ---

    print("\n[4/4] Building wide output with separators...")

    max_len = max(
        len(regionsector_output),
        len(region_output),
        len(sector_output),
        len(sectortype_output),
    )

    rs_padded = pad_df(regionsector_output, max_len)
    region_padded = pad_df(region_output, max_len)
    sector_padded = pad_df(sector_output, max_len)
    stype_padded = pad_df(sectortype_output, max_len)

    sep1 = pd.DataFrame({"sep_1": [np.nan] * max_len})
    sep2 = pd.DataFrame({"sep_2": [np.nan] * max_len})
    sep3 = pd.DataFrame({"sep_3": [np.nan] * max_len})

    final_df = pd.concat(
        [
            rs_padded,          # region_sector + metric
            sep1,
            region_padded,      # region_iso, region_name, avg top3
            sep2,
            sector_padded,      # sector, avg top3
            sep3,
            stype_padded,       # sector_type, avg top100, avg all, n_runs
        ],
        axis=1
    )

    final_df.to_csv(OUT_FILE, index=False)

    print(f"\nDONE. Wide rankings file saved to:\n  {OUT_FILE}")
    print("\nPreview (first 10 rows):")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()