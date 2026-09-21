import pandas as pd
import numpy as np

# --- CONFIG ---
MRIO_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_adjusted.pkl"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"

OUT_FILE_REGION = "total_region_final_demand.csv"
OUT_FILE_REGION_SECTOR_TYPE = "region_sector_type_final_demand_with_shares.csv"

FINAL_DEMAND_LABELS = {
    "Household final consumption P.3h",
    "Non-profit institutions serving households P.3n",
    "Government final consumption P.3g",
    "Gross fixed capital formation P.51",
    "Changes in inventories P.52",
    "Acquisitions less disposals of valuables P.53",
}


def main():
    print("Loading MRIO file...")
    io_df = pd.read_pickle(MRIO_FILE)

    print("Loading sector table...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)

    # Validate structure
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("MRIO row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("MRIO column index must be a MultiIndex with (region, item).")

    row_regions = io_df.index.get_level_values(0).astype(str)
    row_sectors = io_df.index.get_level_values(1).astype(str)
    col_regions = io_df.columns.get_level_values(0).astype(str)
    col_items = io_df.columns.get_level_values(1).astype(str)

    # --- 1. Filter final demand columns ---
    print("Filtering final demand columns...")
    fd_cols_mask = col_items.isin(FINAL_DEMAND_LABELS)
    fd_df = io_df.loc[:, fd_cols_mask]

    if fd_df.shape[1] == 0:
        raise ValueError("No final demand columns matched FINAL_DEMAND_LABELS.")

    # --- 2. Total final demand per destination region (for region_share) ---
    print("Calculating total final demand per region...")
    # Sum all FD entries per consuming region (column level 0)
    total_fd_by_region = (
        fd_df.groupby(level=0, axis=1)
        .sum()            # sum over FD categories
        .sum(axis=0)      # sum over all supplying rows
    )
    total_fd_by_region = total_fd_by_region.rename("total_final_demand").reset_index()
    total_fd_by_region.columns = ["iso", "total_final_demand"]

    region_totals = (
        total_fd_by_region.sort_values("total_final_demand", ascending=False)
        .reset_index(drop=True)
    )

    # --- 3. Build detailed long-form FD with producer & consumer info ---
    print("Building long-form final demand data...")
    # Keep row index explicit
    fd_df = fd_df.copy()
    fd_df.index = pd.MultiIndex.from_arrays([row_regions, row_sectors], names=["producer_iso", "sector"])

    # Make column MultiIndex explicit for melt
    fd_df.columns = pd.MultiIndex.from_arrays([col_regions[fd_cols_mask], col_items[fd_cols_mask]],
                                               names=["iso", "fd_category"])

    # Convert to long: one row per (producer_iso, sector, iso, fd_category)
    fd_long = (
        fd_df.stack(level=[0, 1])
        .reset_index()
    )
    fd_long.columns = ["producer_iso", "sector", "iso", "fd_category", "final_demand"]

    # --- 4. Attach sector type and aggregate by iso, sector_type, producer_iso ---
    print("Mapping sectors to sector types...")
    sector_type_map = sector_table.set_index("sector")["type"]
    fd_long["sector_type"] = fd_long["sector"].map(sector_type_map)

    unmatched = fd_long["sector_type"].isna().sum()
    if unmatched > 0:
        print(f"WARNING: {unmatched} rows have no matched sector_type.")

    print("Aggregating by iso, sector_type, producer_iso...")
    agg = (
        fd_long.groupby(["iso", "sector_type", "producer_iso"], as_index=False)["final_demand"]
        .sum()
    )

    # --- 5. Compute domestic vs foreign shares for each (iso, sector_type) ---
    print("Computing domestic vs foreign shares...")
    # Total FD for this consuming region and sector type
    totals_rst = (
        agg.groupby(["iso", "sector_type"], as_index=False)["final_demand"]
        .sum()
        .rename(columns={"final_demand": "final_demand_total"})
    )

    # Domestic part
    domestic = agg[agg["iso"] == agg["producer_iso"]].copy()
    domestic = (
        domestic.groupby(["iso", "sector_type"], as_index=False)["final_demand"]
        .sum()
        .rename(columns={"final_demand": "final_demand_domestic"})
    )

    # Merge domestic with totals
    rst = totals_rst.merge(domestic, on=["iso", "sector_type"], how="left")
    rst["final_demand_domestic"] = rst["final_demand_domestic"].fillna(0.0)

    # Foreign = total - domestic
    rst["final_demand_foreign"] = rst["final_demand_total"] - rst["final_demand_domestic"]

    # Shares (in %)
    rst["domestic_share"] = np.where(
        rst["final_demand_total"] != 0,
        100.0 * rst["final_demand_domestic"] / rst["final_demand_total"],
        np.nan,
    )
    rst["foreign_share"] = np.where(
        rst["final_demand_total"] != 0,
        100.0 * rst["final_demand_foreign"] / rst["final_demand_total"],
        np.nan,
    )

    # --- 6. Sector-type share in total regional final demand ---
    print("Computing sector-type share in regional final demand...")
    rst = rst.merge(total_fd_by_region, on="iso", how="left")
    rst["region_share"] = np.where(
        rst["total_final_demand"] != 0,
        100.0 * rst["final_demand_total"] / rst["total_final_demand"],
        np.nan,
    )

    # --- 7. Final ordering and export ---
    print("Preparing outputs...")

    rst = (
        rst.sort_values("final_demand_total", ascending=False)
        .reset_index(drop=True)
    )

    # Region-level file
    print(f"Saving {OUT_FILE_REGION}...")
    region_totals.to_csv(OUT_FILE_REGION, index=False)

    # Region–sector-type file with percentages
    print(f"Saving {OUT_FILE_REGION_SECTOR_TYPE}...")
    rst_out = rst[
        [
            "iso",
            "sector_type",
            "final_demand_total",
            "final_demand_domestic",
            "final_demand_foreign",
            "domestic_share",   # % of sector-type FD that is domestic
            "foreign_share",    # % of sector-type FD that is foreign
            "region_share",     # % of region FD accounted for by this sector type
        ]
    ]
    rst_out.to_csv(OUT_FILE_REGION_SECTOR_TYPE, index=False)

    print("Done!")


if __name__ == "__main__":
    main()