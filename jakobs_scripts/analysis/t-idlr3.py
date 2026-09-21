#!/usr/bin/env python3

import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_all.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"
OUT_FILE = "/home/user/Documents/University/Master Thesis/threat_analysis/t-idlr3.csv"

TIME_STEP_THRESHOLD = 12


def compute_loss_metrics(df_loss: pd.DataFrame, disrupted_region_sector: str) -> dict:
    """
    Direct loss:
    - sector == disrupted_region_sector
    - time_step <= TIME_STEP_THRESHOLD

    Indirect loss:
    - total_loss - direct_loss
    """
    time_col = "time_step"

    if "sector" not in df_loss.columns:
        raise ValueError("Column 'sector' not found in loss file. This column must contain the region_sector strings.")

    is_matching_rs = df_loss["sector"] == disrupted_region_sector
    is_early_time = df_loss[time_col] <= TIME_STEP_THRESHOLD
    mask_direct = is_matching_rs & is_early_time

    direct_loss = df_loss.loc[mask_direct, "loss"].sum()
    total_loss = df_loss["loss"].sum()
    indirect_loss = total_loss - direct_loss

    if direct_loss > 0:
        indirect_direct_ratio = indirect_loss / direct_loss
    else:
        indirect_direct_ratio = np.nan

    return {
        "total_loss": total_loss,
        "direct_loss": direct_loss,
        "indirect_loss": indirect_loss,
        "indirect_direct_ratio": indirect_direct_ratio,
    }


def pad_df(df, max_len):
    if len(df) < max_len:
        padding = pd.DataFrame(np.nan, index=np.arange(max_len - len(df)), columns=df.columns)
        return pd.concat([df, padding], ignore_index=True)
    return df


def get_top3_avg_ranking(df, group_col, value_col, label_prefix):
    df_valid = df.dropna(subset=[group_col, value_col])
    if df_valid.empty:
        return pd.DataFrame()

    df_sorted = df_valid.sort_values(by=[group_col, value_col], ascending=[True, False])
    top3_per_group = df_sorted.groupby(group_col, group_keys=False).head(3)

    avg_vals = top3_per_group.groupby(group_col)[value_col].mean().reset_index()
    avg_vals.columns = [group_col, f"{label_prefix}_AvgTop3Ratio"]
    avg_vals = avg_vals.sort_values(by=f"{label_prefix}_AvgTop3Ratio", ascending=False).reset_index(drop=True)
    return avg_vals


def get_topN_avg_per_group(df, group_col, value_col, N):
    df_valid = df.dropna(subset=[group_col, value_col])
    if df_valid.empty:
        return pd.DataFrame()

    df_sorted = df_valid.sort_values(by=[group_col, value_col], ascending=[True, False])
    topN = df_sorted.groupby(group_col, group_keys=False).head(N)
    avg_vals = topN.groupby(group_col)[value_col].mean().reset_index()
    return avg_vals


def main():
    print("=" * 80)
    print("LOSS RANKINGS: GLOBAL LIST + TOP-3 AVG (REGION / SECTOR) + SECTOR-TYPE AVG (TOP100 / ALL)")
    print("=" * 80)

    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]

    if "region_sector" not in runs.columns or "folder_name" not in runs.columns:
        raise ValueError("RUNS_FILE must have 'region_sector' and 'folder_name' columns.")

    runs["region"] = runs["region_sector"].str.split("_", n=1).str[0]
    runs["sector"] = runs["region_sector"].str.split("_", n=1).str[1]
    print(f"  Found {len(runs)} valid simulations")

    print("\n[2/5] Loading sector and region mapping tables...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)
    if "sector" not in sector_table.columns or "type" not in sector_table.columns:
        raise ValueError("SECTOR_TABLE_FILE must contain 'sector' and 'type' columns.")
    sector_type_map = sector_table.set_index("sector")["type"].to_dict()
    runs["sector_type"] = runs["sector"].map(sector_type_map).fillna("Unknown")

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

    region_name_map = dict(zip(regions_df[iso_col].astype(str), regions_df[name_col].astype(str)))

    print("\n[3/5] Processing loss files (calculating per-simulation ratios)...")
    records = []
    missing_count = 0

    for _, row in tqdm(runs.iterrows(), total=len(runs), desc="Processing Simulations"):
        region_sector = row["region_sector"]
        region = row["region"]
        sector = row["sector"]
        sector_type = row["sector_type"]
        folder = row["folder_name"]

        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if loss_path.exists():
            try:
                df_loss = pd.read_csv(loss_path)
                metrics = compute_loss_metrics(df_loss, region_sector)
                del df_loss
                gc.collect()
            except Exception as e:
                tqdm.write(f"  ERROR processing {folder}: {e}")
                metrics = {
                    "total_loss": np.nan,
                    "direct_loss": np.nan,
                    "indirect_loss": np.nan,
                    "indirect_direct_ratio": np.nan,
                }
        else:
            missing_count += 1
            metrics = {
                "total_loss": np.nan,
                "direct_loss": np.nan,
                "indirect_loss": np.nan,
                "indirect_direct_ratio": np.nan,
            }

        records.append({
            "region_sector": region_sector,
            "region": region,
            "sector": sector,
            "sector_type": sector_type,
            "folder": folder,
            "total_loss": metrics["total_loss"],
            "direct_loss": metrics["direct_loss"],
            "indirect_loss": metrics["indirect_loss"],
            "indirect_direct_ratio": metrics["indirect_direct_ratio"],
        })

    print(f"  Completed {len(records)} simulations")
    if missing_count > 0:
        print(f"  WARNING: {missing_count} loss_per_region_sector_time.csv files missing.")

    all_rs = pd.DataFrame(records)
    all_rs["region_name"] = all_rs["region"].astype(str).map(region_name_map).fillna(all_rs["region"])

    print("\n[4/5] Computing rankings...")

    # A) Global ranking, simplified like first script
    global_list = all_rs.dropna(subset=["indirect_direct_ratio"]).copy()
    global_list = global_list.sort_values(by="indirect_direct_ratio", ascending=False).reset_index(drop=True)
    global_output = global_list[["region_sector", "indirect_direct_ratio"]].rename(columns={
        "region_sector": "region_sector",
        "indirect_direct_ratio": "indirect_direct_ratio",
    })

    # B) Region top-3 average ranking with region names
    region_top3 = get_top3_avg_ranking(all_rs, "region", "indirect_direct_ratio", "Region")
    region_top3["region_name"] = region_top3["region"].astype(str).map(region_name_map).fillna(region_top3["region"])
    region_output = region_top3[["region", "region_name", "Region_AvgTop3Ratio"]].rename(columns={
        "region": "region_iso",
        "Region_AvgTop3Ratio": "region_avg_top3_ratio",
    })

    # C) Sector top-3 average ranking
    sector_top3 = get_top3_avg_ranking(all_rs, "sector", "indirect_direct_ratio", "Sector")
    sector_output = sector_top3[["sector", "Sector_AvgTop3Ratio"]].rename(columns={
        "Sector_AvgTop3Ratio": "sector_avg_top3_ratio",
    })

    # D) Sector-type ranking logic changed to match first script style:
    #    - top 100 average
    #    - all-observations average
    #    - occurrence count
    st_valid = all_rs.dropna(subset=["sector_type", "indirect_direct_ratio"]).copy()

    st_top100 = get_topN_avg_per_group(st_valid, "sector_type", "indirect_direct_ratio", 100)
    st_top100 = st_top100.rename(columns={
        "indirect_direct_ratio": "sector_type_avg_top100_ratio",
    })

    st_all = st_valid.groupby("sector_type")["indirect_direct_ratio"].mean().reset_index()
    st_all = st_all.rename(columns={
        "indirect_direct_ratio": "sector_type_avg_all_ratio",
    })

    st_counts = st_valid.groupby("sector_type")["indirect_direct_ratio"].count().reset_index()
    st_counts = st_counts.rename(columns={
        "indirect_direct_ratio": "sector_type_n_runs",
    })

    sectortype_output = (
        st_top100
        .merge(st_all, on="sector_type", how="outer")
        .merge(st_counts, on="sector_type", how="outer")
    )

    sectortype_output = sectortype_output.sort_values(
        by=["sector_type_avg_top100_ratio", "sector_type_avg_all_ratio"],
        ascending=[False, False]
    ).reset_index(drop=True)

    print("\n[5/5] Building wide output with separators...")
    max_len = max(len(global_output), len(region_output), len(sector_output), len(sectortype_output))

    global_padded = pad_df(global_output, max_len)
    region_padded = pad_df(region_output, max_len)
    sector_padded = pad_df(sector_output, max_len)
    stype_padded = pad_df(sectortype_output, max_len)

    sep1 = pd.DataFrame({"sep_1": [np.nan] * max_len})
    sep2 = pd.DataFrame({"sep_2": [np.nan] * max_len})
    sep3 = pd.DataFrame({"sep_3": [np.nan] * max_len})

    final_df = pd.concat([
        global_padded,
        sep1,
        region_padded,
        sep2,
        sector_padded,
        sep3,
        stype_padded,
    ], axis=1)

    final_df.to_csv(OUT_FILE, index=False)

    print(f"\nDONE. Wide rankings file saved to:\n  {OUT_FILE}")
    print("\nPreview (first 10 rows):")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()