import pandas as pd
from pathlib import Path

INPUT_PATH = Path("/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl")
INPUT_FORMAT = "pkl"
OUTPUT_PATH = Path("/home/user/Documents/University/Master Thesis/top_supplier80%.csv")

CHUNK_SIZE = 960  # tune according to RAM


def compute_region_sector_metrics_iterative(io_df: pd.DataFrame, chunk_size: int = 100) -> pd.DataFrame:
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0

    n_cols = io_df.shape[1]
    row_index = io_df.index

    # supplying regions
    row_regions = row_index.get_level_values(row_region_level)

    metric1_sum_shares = pd.Series(0.0, index=row_index)
    metric2_count_pos = pd.Series(0, index=row_index, dtype="int64")
    metric3_value_weighted = pd.Series(0.0, index=row_index)

    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        cols_chunk = io_df.columns[start:end]

        flows_chunk = io_df.loc[:, cols_chunk]

        # column totals over full consumption (including domestic)
        col_totals = flows_chunk.sum(axis=0)

        nonzero = col_totals != 0
        if not nonzero.any():
            continue
        flows_chunk = flows_chunk.loc[:, nonzero]
        col_totals = col_totals[nonzero]
        cols_chunk = cols_chunk[nonzero]

        # shares based on full consumption
        shares_chunk = flows_chunk.div(col_totals, axis=1)

        # mask ONLY for aggregation of metrics, not for col_totals
        col_regions = pd.Index(cols_chunk).get_level_values(col_region_level)
        own_region_mask = pd.DataFrame(
            (row_regions.to_numpy()[:, None] == col_regions.to_numpy()[None, :]),
            index=row_index,
            columns=flows_chunk.columns,
        )


        # masked versions used for metrics (exclude own-region)
        shares_for_metrics = shares_chunk.mask(own_region_mask, 0.0)
        flows_for_metrics = flows_chunk.mask(own_region_mask, 0.0)

        # apply 10% threshold for metrics 1 and 2
        threshold_mask = shares_for_metrics > 0.80
        shares_for_m1_m2 = shares_for_metrics.where(threshold_mask, 0.0)

        # metric 1: sum of shares > ...%
        metric1_sum_shares += shares_for_m1_m2.sum(axis=1)

        # metric 2: count of shares > ...%
        positive_mask = threshold_mask
        metric2_count_pos += positive_mask.sum(axis=1)

        # metric 3: still uses all (non-own-region) flows and shares,
        # with the ...% threshold
        metric3_value_weighted += (flows_for_metrics * shares_for_m1_m2).sum(axis=1)


        # metric 1: sum of masked shares
        #metric1_sum_shares += shares_for_metrics.sum(axis=1)

        # metric 2: count of positive masked shares
        #positive_mask = shares_for_metrics > 0
        #metric2_count_pos += positive_mask.sum(axis=1)

        # metric 3: sum(flow * share) using masked versions
        #metric3_value_weighted += (flows_for_metrics * shares_chunk).sum(axis=1)
        
        # note: shares_chunk still based on full totals
        print(start)

    metric1 = metric1_sum_shares
    with pd.option_context("mode.use_inf_as_na", True):
        metric2 = metric1_sum_shares / metric2_count_pos.replace(0, pd.NA)
    metric3 = metric3_value_weighted

    index_regions = row_index.get_level_values(row_region_level).astype(str)
    index_sectors = row_index.get_level_values(row_sector_level).astype(str)
    region_sector = index_regions + "_" + index_sectors

    metrics_df = pd.DataFrame(
        {
            "region_sector": region_sector,
            "metric1": metric1.values,
            "metric2": metric2.values,
            "metric3": metric3.values,
        }
    )
    return metrics_df



def build_top1000_csv(metrics_df: pd.DataFrame) -> pd.DataFrame:
    m1_sorted = metrics_df.sort_values("metric1", ascending=False)
    m2_sorted = metrics_df.sort_values("metric2", ascending=False)
    m3_sorted = metrics_df.sort_values("metric3", ascending=False)

    m1_top = m1_sorted.head(1000).reset_index(drop=True)
    m2_top = m2_sorted.head(1000).reset_index(drop=True)
    m3_top = m3_sorted.head(1000).reset_index(drop=True)

    max_len = max(len(m1_top), len(m2_top), len(m3_top))

    m1_top = m1_top.reindex(range(max_len))
    m2_top = m2_top.reindex(range(max_len))
    m3_top = m3_top.reindex(range(max_len))

    final_df = pd.DataFrame(
        {
            "region_sector_1": m1_top["region_sector"],
            "metric1": m1_top["metric1"],
            "": ["" for _ in range(max_len)],
            "region_sector_2": m2_top["region_sector"],
            "metric2": m2_top["metric2"],
            " ": ["" for _ in range(max_len)],
            "region_sector_3": m3_top["region_sector"],
            "metric3": m3_top["metric3"],
        }
    )
    return final_df


def main():
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    # exclude last 984 value-added rows
    io_df = io_df.iloc[:-984, :]

    metrics_df = compute_region_sector_metrics_iterative(io_df, chunk_size=CHUNK_SIZE)
    final_df = build_top1000_csv(metrics_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Written {OUTPUT_PATH}")



if __name__ == "__main__":
    main()
