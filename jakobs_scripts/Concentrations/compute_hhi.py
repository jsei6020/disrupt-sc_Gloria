#!/usr/bin/env python3
import pandas as pd

# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"  # "pkl" or "parquet"
OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/sector_hhi_output.csv"
# ========================

def compute_sector_hhi(io_df: pd.DataFrame):
    """
    Memory-efficient HHI of regional output shares for each sector.

    Assumes:
    - Rows: MultiIndex (region_iso3, sector_name)
    - Columns: anything (we only need row sums)
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")

    region_level = 0
    sector_level = 1

    # Row sums (gross output by region-sector) as a Series
    row_sums = io_df.sum(axis=1)

    # Extract sector level once
    sectors = row_sums.index.get_level_values(sector_level)

    # Total output per sector (sum over all regions)
    sector_totals = row_sums.groupby(sectors).sum()  # Series indexed by sector

    # Map sector_total back to each (region, sector) row in a Series
    sector_total_per_row = sector_totals.reindex(sectors).to_numpy()

    # Regional shares per (region, sector)
    shares = row_sums.to_numpy() / sector_total_per_row

    # HHI per sector: sum of squared shares across regions
    share_sq = shares ** 2
    hhi = pd.Series(share_sq, index=row_sums.index).groupby(sectors).sum()
    hhi.name = "HHI"

    # 0–10,000 scaling
    hhi_10000 = (hhi * 10_000).rename("HHI_0_10000")

    # Build result and sort by HHI (descending)
    result = (
        pd.concat([sector_totals.rename("sector_total"), hhi, hhi_10000], axis=1)
          .reset_index()
          .rename(columns={"index": "sector"})
          .sort_values("HHI", ascending=False)
          .reset_index(drop=True)
    )

    return result


def main():
    # Load IO table
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    result = compute_sector_hhi(io_df)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
