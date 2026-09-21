#!/usr/bin/env python3
"""
Combined analysis script that:
1. Filters sectors to target types (agriculture, mining, food, manufacturing, construction)
2. Iteratively computes export values per (sector, region) excluding domestic flows
3. Calculates each sector's export supply share relative to GLOBAL OUTPUT of that sector
4. Identifies the top 3 sectors per region by export supply share
5. Exports a CSV with: iso_region_sector, export_supply_share_pct
"""
import pandas as pd
import numpy as np


# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_adjusted.pkl"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
INPUT_FORMAT = "pkl"  # "pkl" or "csv"

# Output path
TOP3_OUTPUT = "/home/user/Documents/University/Master Thesis/supply_concentrations/top3_filtered_export_sectors_per_region.csv"

# Target sector types from sector_table_g.csv
TARGET_TYPES = {'agriculture', 'mining', 'food', 'manufacturing', 'construction'}
# ========================


def get_allowed_sectors(sector_table_path, target_types):
    """Load sector table and return list of sector names matching target types."""
    df = pd.read_csv(sector_table_path)
    # Handle potential unnamed index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df[df['type'].isin(target_types)]['sector'].tolist()


def compute_sector_region_export_stats(io_df: pd.DataFrame, allowed_sectors: list) -> pd.DataFrame:
    """
    Compute, for every (sector, region), the export supply share relative to global sector output.
    Maintains iterative chunk processing to prevent RAM blowout.
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0

    row_sectors_full = io_df.index.get_level_values(row_sector_level)

    # Filter allowed sectors to only those actually present in the IO table
    io_sectors = row_sectors_full.unique()
    valid_sectors = [s for s in allowed_sectors if s in io_sectors]
    print(f"  Matched {len(valid_sectors)}/{len(allowed_sectors)} target sectors in IO table.")

    records = []
    print(f"Computing sector-region export stats for {len(valid_sectors)} sectors (iterative)...")

    for i, sector_name in enumerate(valid_sectors, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing sector {i}/{len(valid_sectors)}: {sector_name}")

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
        world_total = region_totals.sum()  # Global output for this sector

        if world_total == 0:
            continue

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

        # Shares within this sector: export / world_total_output
        supply_share_exports = region_totals_excl_self / world_total

        # Collect lightweight records
        for reg in region_totals.index:
            records.append(
                {
                    "sector": sector_name,
                    "region": reg,
                    "supply_share_exports": supply_share_exports.loc[reg],
                }
            )

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("TOP 3 FILTERED EXPORT SECTORS PER REGION")
    print("=" * 60)

    # Load sector mapping
    print(f"\nLoading sector table from {SECTOR_TABLE_FILE}...")
    allowed_sectors = get_allowed_sectors(SECTOR_TABLE_FILE, TARGET_TYPES)
    print(f"Target types: {sorted(TARGET_TYPES)}")
    print(f"Number of sectors to include: {len(allowed_sectors)}")

    # Load IO table
    print(f"\nLoading IO table from {INPUT_PATH}...")
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "csv":
        io_df = pd.read_csv(INPUT_PATH, header=[0, 1], index_col=[0, 1])
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'csv'.")
    print(f"Loaded IO table with shape: {io_df.shape}")

    # Step 1: Compute exports iteratively (sector-by-sector)
    print("\n" + "=" * 60)
    print("STEP 1: Computing region-sector export supply shares (global output base)")
    print("=" * 60)
    stats_df = compute_sector_region_export_stats(io_df, allowed_sectors)

    if stats_df.empty:
        print("No export data found for target sectors. Exiting.")
        return

    # Step 2: Convert to percentage, sort, and extract top 3 per region
    print("\n" + "=" * 60)
    print("STEP 2: Calculating top 3 per region")
    print("=" * 60)

    stats_df["export_supply_share_pct"] = (stats_df["supply_share_exports"] * 100).round(2)

    # Sort by region and export supply share (descending)
    stats_df = stats_df.sort_values(
        ["region", "export_supply_share_pct"], ascending=[True, False]
    )

    # Keep only top 3 per region
    top3_df = stats_df.groupby("region").head(3).copy()

    # Format output columns
    top3_df["iso_region_sector"] = (
        top3_df["region"].astype(str) + "_" + top3_df["sector"].astype(str)
    )

    # Export CSV with exactly two columns
    output_df = top3_df[["iso_region_sector", "export_supply_share_pct"]].reset_index(drop=True)
    output_df.to_csv(TOP3_OUTPUT, index=False)

    print(f"\nExported {len(output_df)} region-sector pairs to: {TOP3_OUTPUT}")
    print("\nSample output:")
    print(output_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()