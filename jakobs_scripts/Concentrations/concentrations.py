#!/usr/bin/env python3
import pandas as pd

# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"                 # "pkl" or "parquet"
SECTOR_NAME = "Water transport"         # sector label in the 2nd index level
OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/supply_concentrations/Water_transport_output_shares.csv"
# ========================


def compute_sector_output_shares(io_df: pd.DataFrame, sector_name: str):
    """Compute regional output shares for a given sector from an IO table.

    Row and column index are expected to be MultiIndex: (region_iso3, sector_name).
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    sector_level = 1
    row_region_level = 0
    col_region_level = 0

    # ---- filter rows for the specified sector ----
    row_mask = io_df.index.get_level_values(sector_level) == sector_name
    sector_df = io_df[row_mask]

    if sector_df.empty:
        raise ValueError(f"No rows found for sector '{sector_name}'.")

    # ---- totals including self ----
    row_sums = sector_df.sum(axis=1)
    region_totals = row_sums.groupby(row_sums.index.get_level_values(row_region_level)).sum()
    world_total = region_totals.sum()
    total_supply_share = region_totals / world_total

    # ---- exclude own-region columns ----
    col_regions = sector_df.columns.get_level_values(col_region_level)
    row_regions = sector_df.index.get_level_values(row_region_level)

    same_region_mask = pd.DataFrame(False, index=sector_df.index, columns=sector_df.columns)
    for reg in row_regions.unique():
        same_region_mask.loc[row_regions == reg, col_regions == reg] = True

    sector_excl_self = sector_df.mask(same_region_mask, 0.0)

    row_sums_excl_self = sector_excl_self.sum(axis=1)
    region_totals_excl_self = row_sums_excl_self.groupby(
        row_sums_excl_self.index.get_level_values(row_region_level)
    ).sum()

    supply_share_exports = region_totals_excl_self / world_total

    # ---- build result, sorted by share excluding self ----
    result = pd.DataFrame({
        "region": region_totals.index,
        "total_supply_share": total_supply_share.values,
        "total_output": region_totals.values,
        "~net export": region_totals_excl_self.reindex(region_totals.index).values,
        "supply_share_exports": supply_share_exports.reindex(region_totals.index).values,
    })

    result = result.sort_values("supply_share_exports", ascending=False).reset_index(drop=True)
    return result


def main():
    # Load IO table
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    # Compute and export
    result = compute_sector_output_shares(io_df, SECTOR_NAME)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
