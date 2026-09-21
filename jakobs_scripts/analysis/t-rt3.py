#!/usr/bin/env python3

import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_all.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"

OUT_FILE = "/home/user/Documents/University/Master Thesis/threat_analysis/t-rt3.csv"

# Thousands USD threshold for 90% metric is implicit via total loss,
# so no explicit THRESHOLD here – we only use the time_to_90pct metric.


def compute_time_to_90pct(df_loss: pd.DataFrame) -> float:
    """
    Compute time_to_90pct: first timestep when cumulative global losses
    (summed over all region-sectors) reach 90% of total loss.

    Interpretation:
      - Lower time_to_90pct  => faster damage accumulation => less resilient
      - Higher time_to_90pct => slower accumulation       => more resilient
    """
    loss_by_time = df_loss.groupby("time_step")["loss"].sum().reset_index()
    loss_by_time = loss_by_time.sort_values("time_step")

    loss_by_time["cumulative_loss"] = loss_by_time["loss"].cumsum()
    total_loss = loss_by_time["cumulative_loss"].max()

    if total_loss <= 0 or pd.isna(total_loss):
        return np.nan

    threshold_90 = total_loss * 0.9
    mask_90 = loss_by_time["cumulative_loss"] >= threshold_90

    if not mask_90.any():
        return np.nan

    return float(loss_by_time.loc[mask_90, "time_step"].min())


def pad_df(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    if len(df) < max_len:
        padding = pd.DataFrame(np.nan, index=np.arange(max_len - len(df)), columns=df.columns)
        return pd.concat([df, padding], ignore_index=True)
    return df


def get_top3_avg_ranking(df: pd.DataFrame, group_col: str, value_col: str, label_prefix: str) -> pd.DataFrame:
    """
    For each group:
      - sort by value_col descending (we want the longest recovery times)
      - take top 3 rows
      - compute mean of those top 3
      - rank groups by this mean (descending)
    """
    df_valid = df.dropna(subset=[group_col, value_col])
    if df_valid.empty:
        return pd.DataFrame()

    df_sorted = df_valid.sort_values(by=[group_col, value_col], ascending=[True, False])
    top3_per_group = df_sorted.groupby(group_col, group_keys=False).head(3)

    avg_vals = top3_per_group.groupby(group_col)[value_col].mean().reset_index()
    avg_vals.columns = [group_col, f"{label_prefix}_AvgTop3TimeTo90pct"]

    avg_vals = avg_vals.sort_values(
        by=f"{label_prefix}_AvgTop3TimeTo90pct", ascending=False
    ).reset_index(drop=True)

    return avg_vals


def get_topN_avg_per_group(df: pd.DataFrame, group_col: str, value_col: str, N: int) -> pd.DataFrame:
    """
    For each group:
      - sort by value_col descending
      - take top N rows
      - compute mean of those top N values
    """
    df_valid = df.dropna(subset=[group_col, value_col])
    if df_valid.empty:
        return pd.DataFrame()

    df_sorted = df_valid.sort_values(by=[group_col, value_col], ascending=[True, False])
    topN = df_sorted.groupby(group_col, group_keys=False).head(N)
    avg_vals = topN.groupby(group_col)[value_col].mean().reset_index()
    return avg_vals


def main():
    print("=" * 80)
    print("TIME TO 90% DAMAGE: GLOBAL + TOP-3 AVG (REGION / SECTOR) + SECTOR-TYPE AVG (TOP100 / ALL)")
    print("=" * 80)

    # --- 1. Load runs, sector types, region mapping ---
    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]

    if "region_sector" not in runs.columns or "folder_name" not in runs.columns:
        raise ValueError("RUNS_FILE must have 'region_sector' and 'folder_name' columns.")

    runs["region"] = runs["region_sector"].str.split("_", n=1).str[0]
    runs["sector"] = runs["region_sector"].str.split("_", n=1).str[1]
    print(f"  Found {len(runs)} valid simulations")

    print("\n[2/5] Loading sector table for sector_type mapping...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)
    if "sector" not in sector_table.columns or "type" not in sector_table.columns:
        raise ValueError("SECTOR_TABLE_FILE must contain 'sector' and 'type' columns.")
    sector_type_map: Dict[str, str] = sector_table.set_index("sector")["type"].to_dict()
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

    # --- 3. Loop over runs and compute time_to_90pct per simulation ---
    print("\n[3/5] Processing loss files (one folder per simulation)...")

    records = []
    missing_count = 0

    for idx, row in tqdm(runs.iterrows(), total=len(runs), desc="Processing Simulations"):
        region_sector = row["region_sector"]
        region = row["region"]
        sector = row["sector"]
        sector_type = row["sector_type"]
        folder = row["folder_name"]

        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if loss_path.exists():
            try:
                df_loss = pd.read_csv(loss_path)
                time_to_90pct = compute_time_to_90pct(df_loss)
                del df_loss
                gc.collect()
            except Exception as e:
                print(f"  ERROR processing {folder}: {e}")
                time_to_90pct = np.nan
        else:
            print(f"  WARNING: {folder} - loss_per_region_sector_time.csv not found")
            missing_count += 1
            time_to_90pct = np.nan

        records.append(
            {
                "region_sector": region_sector,
                "region": region,
                "sector": sector,
                "sector_type": sector_type,
                "folder": folder,
                "time_to_90pct": time_to_90pct,
            }
        )

    print(f"\n  Completed {len(records)} simulations")
    if missing_count > 0:
        print(f"  WARNING: {missing_count} loss_per_region_sector_time.csv files missing.")

    all_rs = pd.DataFrame(records)
    all_rs["region_name"] = (
        all_rs["region"].astype(str).map(region_name_map).fillna(all_rs["region"])
    )

    metric_col = "time_to_90pct"

    # --- 4. Compute rankings / averages ---
    print("\n[4/5] Computing rankings...")

    # 4A. Global region-sector ranking: just region_sector + metric
    rs_df = all_rs[["region_sector", metric_col]].copy()
    rs_df = rs_df.dropna(subset=[metric_col])
    # Sort descending: longest recovery times first (more resilient)
    rs_df = rs_df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    regionsector_output = rs_df.rename(
        columns={metric_col: "region_sector_time_to_90pct"}
    )

    # 4B. Region top-3 average ranking (with region_name)
    region_top3 = get_top3_avg_ranking(
        all_rs, group_col="region", value_col=metric_col, label_prefix="Region"
    )
    region_top3["region_name"] = (
        region_top3["region"].astype(str).map(region_name_map).fillna(region_top3["region"])
    )

    region_output = region_top3[
        ["region", "region_name", "Region_AvgTop3TimeTo90pct"]
    ].rename(
        columns={
            "region": "region_iso",
            "Region_AvgTop3TimeTo90pct": "region_avg_top3_time_to_90pct",
        }
    )

    # 4C. Sector top-3 average ranking
    sector_top3 = get_top3_avg_ranking(
        all_rs, group_col="sector", value_col=metric_col, label_prefix="Sector"
    )

    sector_output = sector_top3[["sector", "Sector_AvgTop3TimeTo90pct"]].rename(
        columns={
            "Sector_AvgTop3TimeTo90pct": "sector_avg_top3_time_to_90pct",
        }
    )

    # 4D. Sector-type averages:
    #   - top 100 time_to_90pct per sector_type (descending)
    #   - all observations per sector_type
    #   - number of observations per sector_type
    st_valid = all_rs.dropna(subset=["sector_type", metric_col])

    # Top-100 per sector_type (longest times)
    st_top100 = get_topN_avg_per_group(
        st_valid, group_col="sector_type", value_col=metric_col, N=100
    )
    st_top100 = st_top100.rename(
        columns={metric_col: "sector_type_avg_top100_time_to_90pct"}
    )

    # All observations per sector_type
    st_all = st_valid.groupby("sector_type")[metric_col].mean().reset_index()
    st_all = st_all.rename(
        columns={metric_col: "sector_type_avg_all_time_to_90pct"}
    )

    # Count of observations per sector_type
    st_counts = st_valid.groupby("sector_type")[metric_col].count().reset_index()
    st_counts = st_counts.rename(
        columns={metric_col: "sector_type_n_runs"}
    )

    sectortype_output = (
        st_top100.merge(st_all, on="sector_type", how="outer")
        .merge(st_counts, on="sector_type", how="outer")
    )

    sectortype_output = sectortype_output.sort_values(
        by=["sector_type_avg_top100_time_to_90pct", "sector_type_avg_all_time_to_90pct"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # --- 5. Combine into one wide file with separator columns ---
    print("\n[5/5] Building wide output with separators...")

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
            rs_padded,      # region_sector + time_to_90pct
            sep1,
            region_padded,  # region_iso, region_name, avg top3 time_to_90pct
            sep2,
            sector_padded,  # sector, avg top3 time_to_90pct
            sep3,
            stype_padded,   # sector_type, top100 avg, all avg, n_runs
        ],
        axis=1,
    )

    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUT_FILE, index=False)

    print(f"\nDONE. Wide rankings file saved to:\n  {OUT_FILE}")
    print("\nPreview (first 10 rows):")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()