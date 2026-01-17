#!/usr/bin/env python3
import pandas as pd


# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"                 # "pkl" or "parquet"

OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/hhi_excl_self_all_region_sector.csv"
# ========================


def compute_hhi_excl_self_all_pairs(io_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HHI of import concentration for every (consuming_region, sector) pair,
    excluding own-region supply.

    Row index: MultiIndex (supplying_region, sector)
    Column index: MultiIndex (consuming_region, sector)

    Returns a DataFrame with columns:
      - consuming_region
      - sector
      - hhi_excl_self
      - n_suppliers_excl_self
      - total_import_excl_self
    """

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0
    col_sector_level = 1

    results = []

    col_index = io_df.columns
    col_regions = col_index.get_level_values(col_region_level)
    col_sectors = col_index.get_level_values(col_sector_level)

    for col_pos, (cons_region, cons_sector) in enumerate(zip(col_regions, col_sectors)):
        # Take this column as a Series indexed by (supplying_region, sector)
        col_series = io_df.iloc[:, col_pos]

        # Filter rows to the same sector as the consuming sector
        row_sectors = col_series.index.get_level_values(row_sector_level)
        sector_mask = row_sectors == cons_sector
        sector_series = col_series[sector_mask]

        if sector_series.empty:
            continue

        # Exclude own-region supply
        supplying_regions = sector_series.index.get_level_values(row_region_level)
        non_self_mask = supplying_regions != cons_region
        sector_series_excl_self = sector_series[non_self_mask]

        total_import_excl_self = sector_series_excl_self.sum()
        if total_import_excl_self <= 0:
            # No meaningful external supply, set HHI to NaN and continue
            hhi = float("nan")
            n_suppliers = 0
        else:
            # Aggregate by supplying region
            region_supply = sector_series_excl_self.groupby(supplying_regions[non_self_mask]).sum()
            shares = region_supply / total_import_excl_self
            hhi = (shares ** 2).sum()
            n_suppliers = shares.size

        results.append({
            "consuming_region": cons_region,
            "sector": cons_sector,
            "hhi_excl_self": hhi,
            "n_suppliers_excl_self": n_suppliers,
            "total_import_excl_self": total_import_excl_self,
        })
        print(cons_region, cons_sector)

    result_df = pd.DataFrame(results)

    # Sort by HHI descending (NaNs last)
    result_df = result_df.sort_values("hhi_excl_self", ascending=False, na_position="last").reset_index(drop=True)
    return result_df


def main():
    # Load IO table
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    hhi_df = compute_hhi_excl_self_all_pairs(io_df)

    hhi_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
