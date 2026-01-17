#!/usr/bin/env python3
import pandas as pd

# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"  # "pkl" or "parquet"
OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/sector_hhi_net_export_share.csv"
# ========================


def compute_hhi_net_export_by_sector(io_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector, compute the HHI of net-export shares, where
    net export per region is approximated as row sums excluding
    own-region columns (exports to rest of world).

    Assumes:
    - Rows: MultiIndex (region_iso3, sector_name)
    - Columns: MultiIndex (region_iso3, sector_name)
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0

    sectors = io_df.index.get_level_values(row_sector_level).unique()
    results = []

    for sector_name in sectors:
        # ---- filter rows for this sector ----
        row_mask = io_df.index.get_level_values(row_sector_level) == sector_name
        sector_df = io_df[row_mask]

        if sector_df.empty:
            continue

        # ---- totals including self (total supply per region) ----
        row_sums = sector_df.sum(axis=1)
        region_totals = row_sums.groupby(
            row_sums.index.get_level_values(row_region_level)
        ).sum()
        world_total = region_totals.sum()

        # ---- exclude own-region columns: exports to rest of world ----
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

        # net-export share per region: exports (excl self) / world_total supply
        net_export_share = region_totals_excl_self / world_total

        # HHI over these net-export shares
        hhi_net_export_share = (net_export_share ** 2).sum()

        results.append(
            {
                "sector": sector_name,
                "HHI_net_export_share": hhi_net_export_share,
            }
        )
        print(sector_name)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("HHI_net_export_share", ascending=False).reset_index(drop=True)
    return result_df


def main():
    # Load IO table
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    result = compute_hhi_net_export_by_sector(io_df)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
