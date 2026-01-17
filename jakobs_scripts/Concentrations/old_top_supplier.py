#!/usr/bin/env python3
import pandas as pd


# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"                 # "pkl" or "parquet"

OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/top_external_supplier_stats_with_sectors.csv"
# ========================


def compute_top_external_supplier_stats(io_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (consuming_region, sector) column:
      - restrict to supplying rows with the same sector
      - exclude own-region supply
      - find the top external supplier and its share in local consumption
    Then aggregate by supplying region:
      - count how many times it is the top external supplier
      - sum of its shares when it is the top external supplier.

    Row index: MultiIndex (supplying_region, sector)
    Column index: MultiIndex (consuming_region, sector)
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

    # Collect all (cons_region, cons_sector) -> (top_supplier_region, share)
    records = []

    for col_pos, (cons_region, cons_sector) in enumerate(zip(col_regions, col_sectors)):
        col_series = io_df.iloc[:, col_pos]  # Series indexed by (supp_region, sector)

        row_sectors = col_series.index.get_level_values(row_sector_level)
        same_sector_mask = row_sectors == cons_sector
        sector_series = col_series[same_sector_mask]

        if sector_series.empty:
            continue

        supp_regions = sector_series.index.get_level_values(row_region_level)
        non_self_mask = supp_regions != cons_region
        external_series = sector_series[non_self_mask]

        total_local_consumption = sector_series.sum()
        if total_local_consumption <= 0:
            continue

        if external_series.empty:
            # No external supplier: skip
            continue

        # Aggregate by supplying region
        external_supply_by_region = external_series.groupby(
            supp_regions[non_self_mask]
        ).sum()

        # Identify top external supplier and its share in local consumption
        top_region = external_supply_by_region.idxmax()
        top_supply = external_supply_by_region.loc[top_region]
        top_share_in_local = top_supply / total_local_consumption

        # Debug print if desired
        print(cons_region, cons_sector)

        records.append({
            "consuming_region": cons_region,
            "sector": cons_sector,
            "top_external_supplier": top_region,
            "top_external_share_in_local": top_share_in_local,
        })

    if not records:
        return pd.DataFrame(columns=[
            "region",
            "n_times_top_external_supplier",
            "sum_top_external_shares_in_local"
        ])

    df_pairs = pd.DataFrame(records)

        # ---- aggregate by supplying region (basic stats) ----
    stats = df_pairs.groupby("top_external_supplier").agg(
        n_times_top_external_supplier=("top_external_share_in_local", "size"),
        sum_top_external_shares_in_local=("top_external_share_in_local", "sum"),
    )
    stats = stats.reset_index().rename(columns={"top_external_supplier": "region"})

    # ---- compute top 10 sectors per supplying region by frequency ----
    counts = df_pairs.groupby(["top_external_supplier", "sector"]).size().reset_index(name="count")
    counts = counts.sort_values(["top_external_supplier", "count"], ascending=[True, False])

    counts["rank"] = counts.groupby("top_external_supplier")["count"].rank(
        method="first", ascending=False
    )
    top_counts = counts[counts["rank"] <= 10].copy()

    # build wide table: one row per top_external_supplier, columns sector_1..sector_10
    # first, sort by supplier and rank
    top_counts = top_counts.sort_values(["top_external_supplier", "rank"])

    # assign position 1..10 per supplier (rank is already that, but ensure int)
    top_counts["pos"] = top_counts["rank"].astype(int)

    # pivot sectors into columns
    sectors_wide = (
        top_counts
        .pivot(index="top_external_supplier", columns="pos", values="sector")
        .reset_index()
    )

    # rename columns: 1 -> sector_1, ..., 10 -> sector_10
    sectors_wide = sectors_wide.rename(
        columns={"top_external_supplier": "region"}
    )
    sectors_wide.columns = [
        "region" if c == "region" else f"sector_{c}" for c in sectors_wide.columns
    ]

    # ---- merge stats and sector list ----
    summary = pd.merge(stats, sectors_wide, on="region", how="left")

    # Sort by sum of shares descending (or by count)
    summary = summary.sort_values(
        ["sum_top_external_shares_in_local", "n_times_top_external_supplier"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return summary


def main():
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    result = compute_top_external_supplier_stats(io_df)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
