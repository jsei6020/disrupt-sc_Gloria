#!/usr/bin/env python3
import pandas as pd


# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"                 # "pkl" or "parquet"

SECTOR_NAME = "Growing rice"         # sector in the 2nd index level
CONSUMING_REGION = "AGO"             # ISO3 of consuming region, e.g. Nigeria

OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/growing_rice_NGA_concentrations.csv"
# ========================


def compute_import_concentrations(io_df: pd.DataFrame,
                                  sector_name: str,
                                  consuming_region: str) -> pd.DataFrame:
    """
    For a given consuming region and sector, compute:
      - regional supply to that consuming region
      - concentration = regional_supply / total_supply_to_region

    Output: rows = supplying regions, columns = [concentration_share, region_supply_to_region]
    """

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0
    col_sector_level = 1

    col_regions = io_df.columns.get_level_values(col_region_level)
    col_sectors = io_df.columns.get_level_values(col_sector_level)

    col_mask = (col_regions == consuming_region) & (col_sectors == sector_name)
    if not col_mask.any():
        raise ValueError(
            f"No column found for consuming region '{consuming_region}' and sector '{sector_name}'."
        )

    col_series = io_df.loc[:, col_mask].iloc[:, 0]

    row_sectors = col_series.index.get_level_values(row_sector_level)
    sector_mask = row_sectors == sector_name
    sector_series = col_series[sector_mask]

    if sector_series.empty:
        raise ValueError(
            f"No supplying rows found for sector '{sector_name}' for consuming region '{consuming_region}'."
        )

    supplying_regions = sector_series.index.get_level_values(row_region_level)
    region_supply = sector_series.groupby(supplying_regions).sum()

    total_supply_to_region = region_supply.sum()
    concentration = region_supply / total_supply_to_region

    # Build *vertical* table and sort by concentration_share descending
    result = pd.DataFrame({
        "supplying_region": region_supply.index,
        "concentration_share": concentration.values,
        "region_supply_to_region": region_supply.values,
    })

    result = result.sort_values("concentration_share", ascending=False).reset_index(drop=True)
    return result


def main():
    # Load IO table
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    result = compute_import_concentrations(io_df, SECTOR_NAME, CONSUMING_REGION)

    # result: rows = [supplying_region, concentration_share, region_supply_to_region]
    # columns = ISO3 of supplying regions
    result.to_csv(OUTPUT_PATH, index=True, header=False)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
