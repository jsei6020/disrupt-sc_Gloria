#!/usr/bin/env python3
"""
Combined analysis script that:
1. Computes HHI of net-export shares for all sectors
2. Identifies regions with >1% export share per sector
3. Exports two CSV files with the results

CSV 1: region_sector_pairs_above_1pct.csv
    - Single column region_sector (e.g. USA_Water transport)

CSV 2: sector_concentration_summary.csv
    - One row per sector
    - Columns: sector, HHI_net_export_share,
               included_countries, sum_supply_share_exports
"""
import pandas as pd
import numpy as np


# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl" #mrio_va_fd.pkl
INPUT_FORMAT = "pkl"  # "pkl" or "csv"

# Output paths
HHI_OUTPUT = "/home/user/Documents/University/Master Thesis/sector_hhi_net_export_share.csv"
REGION_SECTOR_PAIRS_OUTPUT = "/home/user/Documents/University/Master Thesis/supply_concentrations/region_sector_pairs_1pct.csv"
SECTOR_SUMMARY_OUTPUT = "/home/user/Documents/University/Master Thesis/supply_concentrations/sector_concentration_summary.csv"

# Threshold for export share (1%)
EXPORT_SHARE_THRESHOLD = 0.01

# Optional: Filter to analyze only sectors above a certain HHI threshold
# Set to None to analyze all sectors
HHI_THRESHOLD = 0.00167
# ========================


def compute_sector_region_export_stats(io_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute, for every (sector, region), the following quantities:

    - total_output: total supply including domestic
    - total_supply_share: total_output / world_total (within sector)
    - ~net export: exports to rest of world (excluding domestic use)
    - supply_share_exports: (~net export) / world_total (within sector)

    Assumes:
    - Rows: MultiIndex (region_iso3, sector_name)
    - Columns: MultiIndex (region_iso3, sector_name)

    Returns DataFrame with columns:
        sector, region, total_output, total_supply_share,
        ~net export, supply_share_exports
    and one row per (sector, region).
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0

    row_regions_full = io_df.index.get_level_values(row_region_level)
    row_sectors_full = io_df.index.get_level_values(row_sector_level)

    sectors = row_sectors_full.unique()
    records = []

    print(f"Computing sector-region export stats for {len(sectors)} sectors...")

    for i, sector_name in enumerate(sectors, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing sector {i}/{len(sectors)}: {sector_name}")

        # Subset rows for this sector
        row_mask = row_sectors_full == sector_name
        sector_df = io_df[row_mask]

        if sector_df.empty:
            continue

        # ---- Total supply including domestic ----
        row_sums = sector_df.sum(axis=1)  # index: (region, sector)
        region_totals = row_sums.groupby(
            row_sums.index.get_level_values(row_region_level)
        ).sum()
        world_total = region_totals.sum()

        # ---- Exclude own-region columns to get exports to rest of world ----
        sub_row_regions = sector_df.index.get_level_values(row_region_level)
        sub_col_regions = sector_df.columns.get_level_values(col_region_level)

        same_region_mask = pd.DataFrame(
            False, index=sector_df.index, columns=sector_df.columns
        )
        for reg in sub_row_regions.unique():
            same_region_mask.loc[sub_row_regions == reg, sub_col_regions == reg] = True

        sector_excl_self = sector_df.mask(same_region_mask, 0.0)

        row_sums_excl_self = sector_excl_self.sum(axis=1)
        region_totals_excl_self = row_sums_excl_self.groupby(
            row_sums_excl_self.index.get_level_values(row_region_level)
        ).sum()

        # Ensure alignment
        region_totals_excl_self = region_totals_excl_self.reindex(
            region_totals.index
        ).fillna(0.0)

        # Shares within this sector
        total_supply_share = region_totals / world_total
        supply_share_exports = region_totals_excl_self / world_total

        # Collect records
        for reg in region_totals.index:
            records.append(
                {
                    "sector": sector_name,
                    "region": reg,
                    "total_output": region_totals.loc[reg],
                    "total_supply_share": total_supply_share.loc[reg],
                    "~net export": region_totals_excl_self.loc[reg],
                    "supply_share_exports": supply_share_exports.loc[reg],
                }
            )

    result = pd.DataFrame(records)
    return result


def main():
    print("=" * 60)
    print("COMBINED SECTOR CONCENTRATION ANALYSIS")
    print("=" * 60)

    # Load IO table
    print(f"\nLoading IO table from {INPUT_PATH}...")
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "csv":
        # First two rows = column MultiIndex, first two columns = row MultiIndex [web:84][web:94]
        io_df = pd.read_csv(
            INPUT_PATH,
            header=[0, 1],      # two header rows -> MultiIndex columns
            index_col=[0, 1],   # first two columns -> MultiIndex index
        )
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'csv'.")

    print(f"Loaded IO table with shape: {io_df.shape}")

    # Step 1: Compute per-sector, per-region export stats
    print("\n" + "=" * 60)
    print("STEP 1: Computing sector-region export stats")
    print("=" * 60)

    stats_df = compute_sector_region_export_stats(io_df)

    # Step 2: Compute sector-level HHI from regional export shares
    print("\n" + "=" * 60)
    print("STEP 2: Computing sector-level HHI from export shares")
    print("=" * 60)

    # HHI = sum over regions of (supply_share_exports^2) per sector [web:36]
    hhi_df = (
        stats_df
        .groupby("sector")["supply_share_exports"]
        .apply(lambda s: (s ** 2).sum())
        .reset_index(name="HHI_net_export_share")
        .sort_values("HHI_net_export_share", ascending=False)
        .reset_index(drop=True)
    )
    hhi_df.to_csv(HHI_OUTPUT, index=False)
    print(f"\nHHI results saved to: {HHI_OUTPUT}")
    print("\nTop 10 sectors by HHI:")
    print(hhi_df.head(10).to_string(index=False))

    # Step 3: Apply HHI threshold to select sectors
    if HHI_THRESHOLD is not None:
        sectors_to_use = hhi_df[
            hhi_df["HHI_net_export_share"] >= HHI_THRESHOLD
        ]["sector"].tolist()
        print(f"\nRestricting to {len(sectors_to_use)} sectors with HHI >= {HHI_THRESHOLD}")
    else:
        sectors_to_use = hhi_df["sector"].tolist()
        print(f"\nUsing all {len(sectors_to_use)} sectors")

    stats_filtered = stats_df[stats_df["sector"].isin(sectors_to_use)].copy()

    # Step 4: Build CSV 1: only region_sector for pairs above threshold
    print("\n" + "=" * 60)
    print(f"STEP 3: Building CSV 1 (region_sector, threshold {EXPORT_SHARE_THRESHOLD*100}%)")
    print("=" * 60)

    above_thresh = stats_filtered[
        stats_filtered["supply_share_exports"] >= EXPORT_SHARE_THRESHOLD
    ].copy()

    if not above_thresh.empty:
        above_thresh["region_sector"] = (
            above_thresh["region"].astype(str)
            + "_"
            + above_thresh["sector"].astype(str)
        )

        csv1_df = above_thresh[["region_sector"]].copy().reset_index(drop=True)

        csv1_df.to_csv(REGION_SECTOR_PAIRS_OUTPUT, index=False)
        print(f"\nExported {len(csv1_df)} region_sector entries to:")
        print(f"  {REGION_SECTOR_PAIRS_OUTPUT}")
        print("\nSample of CSV 1:")
        print(csv1_df.head(10).to_string(index=False))
    else:
        print("\nNo region_sector pairs above threshold.")
        csv1_df = pd.DataFrame(columns=["region_sector"])

    # Step 5: Build CSV 2: one row per sector, with HHI, included countries, and sum of their export shares
    print("\n" + "=" * 60)
    print("STEP 4: Building CSV 2 (sector summary)")
    print("=" * 60)

    if not above_thresh.empty:
        # Map HHI to sectors
        hhi_map = hhi_df.set_index("sector")["HHI_net_export_share"]

        # Group by sector over the subset above threshold
        summary = (
            above_thresh
            .groupby("sector")
            .agg(
                included_countries=("region", lambda r: ", ".join(sorted(r.unique()))),
                sum_supply_share_exports=("supply_share_exports", "sum"),
            )
            .reset_index()
        )

        summary["HHI_net_export_share"] = summary["sector"].map(hhi_map)

        # Reorder columns
        summary = summary[
            [
                "sector",
                "HHI_net_export_share",
                "included_countries",
                "sum_supply_share_exports",
            ]
        ].sort_values("HHI_net_export_share", ascending=False).reset_index(drop=True)

        summary.to_csv(SECTOR_SUMMARY_OUTPUT, index=False)
        print(f"\nExported {len(summary)} sector summaries to:")
        print(f"  {SECTOR_SUMMARY_OUTPUT}")
        print("\nTop 10 rows of CSV 2:")
        print(summary.head(10).to_string(index=False))
    else:
        print("\nNo sector summaries to export (no pairs above threshold).")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
