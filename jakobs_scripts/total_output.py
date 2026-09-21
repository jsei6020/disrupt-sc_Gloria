#!/usr/bin/env python3
import gc
import numpy as np
import pandas as pd


# --- CONFIG ---
MRIO_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_adjusted.pkl"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"

OUT_FILE_REGION = "/home/user/Documents/University/Master Thesis/total_region_output_with_shares.csv"
OUT_FILE_REGION_SECTOR_TYPE = "/home/user/Documents/University/Master Thesis/region_sector_type_total_output_with_shares.csv"

CHUNK_SIZE = 500  # lower to 250 or 100 if RAM is still tight


def compute_output_shares_by_sector_type_chunked(
    io_df: pd.DataFrame,
    sector_table: pd.DataFrame,
    chunk_size: int = 500
):
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("MRIO row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("MRIO column index must be a MultiIndex with (region, use).")
    if "sector" not in sector_table.columns or "type" not in sector_table.columns:
        raise ValueError("Sector table must contain columns 'sector' and 'type'.")

    # sector -> sector_type mapping
    sector_type_map = (
        sector_table
        .drop_duplicates(subset=["sector"])
        .set_index("sector")["type"]
    )
    valid_sectors = set(sector_type_map.index.astype(str))

    # keep only normal sector rows, drop VA / balancing rows
    row_sector_names = io_df.index.get_level_values(1).astype(str)
    keep_mask = row_sector_names.isin(valid_sectors)
    io_df = io_df.loc[keep_mask]

    col_regions = pd.Index(io_df.columns.get_level_values(0).astype(str))

    region_acc = {}
    region_sector_type_acc = {}

    n_rows = len(io_df)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = io_df.iloc[start:end]

        row_regions = pd.Index(chunk.index.get_level_values(0).astype(str))
        row_sectors = pd.Index(chunk.index.get_level_values(1).astype(str))

        # total output per origin row
        total_output = chunk.sum(axis=1)

        # domestic output per origin row:
        # for each origin region in this chunk, sum only columns with same destination region
        domestic_parts = []
        for reg in row_regions.unique():
            row_mask = (row_regions == reg)
            domestic_col_mask = (col_regions == reg)
            part = chunk.iloc[row_mask, domestic_col_mask].sum(axis=1)
            domestic_parts.append(part)

        domestic_output = pd.concat(domestic_parts).reindex(chunk.index)
        foreign_output = total_output - domestic_output

        df_chunk = pd.DataFrame({
            "iso": row_regions.to_numpy(),
            "sector": row_sectors.to_numpy(),
            "total_output": total_output.to_numpy(),
            "domestic_output": domestic_output.to_numpy(),
            "foreign_output": foreign_output.to_numpy(),
        })

        df_chunk["sector_type"] = df_chunk["sector"].map(sector_type_map)

        if df_chunk["sector_type"].isna().any():
            missing = sorted(df_chunk.loc[df_chunk["sector_type"].isna(), "sector"].unique().tolist())[:20]
            raise ValueError(f"Unexpected unmapped sectors remain after filtering: {missing}")

        # aggregate region totals for this chunk
        region_tmp = (
            df_chunk
            .groupby("iso", as_index=False)[["total_output", "domestic_output", "foreign_output"]]
            .sum()
        )

        for row in region_tmp.itertuples(index=False):
            iso = row.iso
            if iso not in region_acc:
                region_acc[iso] = {
                    "total_output": 0.0,
                    "domestic_output": 0.0,
                    "foreign_output": 0.0
                }
            region_acc[iso]["total_output"] += float(row.total_output)
            region_acc[iso]["domestic_output"] += float(row.domestic_output)
            region_acc[iso]["foreign_output"] += float(row.foreign_output)

        # aggregate region-sector_type totals for this chunk
        rst_tmp = (
            df_chunk
            .groupby(["iso", "sector_type"], as_index=False)[["total_output", "domestic_output", "foreign_output"]]
            .sum()
        )

        for row in rst_tmp.itertuples(index=False):
            key = (row.iso, row.sector_type)
            if key not in region_sector_type_acc:
                region_sector_type_acc[key] = {
                    "total_output": 0.0,
                    "domestic_output": 0.0,
                    "foreign_output": 0.0
                }
            region_sector_type_acc[key]["total_output"] += float(row.total_output)
            region_sector_type_acc[key]["domestic_output"] += float(row.domestic_output)
            region_sector_type_acc[key]["foreign_output"] += float(row.foreign_output)

        print(f"Processed rows {start}:{end} / {n_rows}")

        del chunk, total_output, domestic_parts, domestic_output, foreign_output, df_chunk, region_tmp, rst_tmp
        gc.collect()

    # final region table
    region_totals = pd.DataFrame([
        {
            "iso": iso,
            "total_output": vals["total_output"],
            "domestic_output": vals["domestic_output"],
            "foreign_output": vals["foreign_output"],
        }
        for iso, vals in region_acc.items()
    ])

    region_totals["domestic_share"] = np.where(
        region_totals["total_output"] != 0,
        region_totals["domestic_output"] / region_totals["total_output"],
        np.nan,
    )
    region_totals["foreign_share"] = np.where(
        region_totals["total_output"] != 0,
        region_totals["foreign_output"] / region_totals["total_output"],
        np.nan,
    )

    region_totals = (
        region_totals
        .sort_values("total_output", ascending=False)
        .reset_index(drop=True)
    )

    region_totals = region_totals[
        ["iso", "total_output", "domestic_output", "foreign_output", "domestic_share", "foreign_share"]
    ]

    # final region-sector_type table
    rst_totals = pd.DataFrame([
        {
            "iso": iso,
            "sector_type": sector_type,
            "total_output_total": vals["total_output"],
            "total_output_domestic": vals["domestic_output"],
            "total_output_foreign": vals["foreign_output"],
        }
        for (iso, sector_type), vals in region_sector_type_acc.items()
    ])

    # shares within each region-sector_type
    rst_totals["domestic_share"] = np.where(
        rst_totals["total_output_total"] != 0,
        100.0 * rst_totals["total_output_domestic"] / rst_totals["total_output_total"],
        np.nan,
    )
    rst_totals["foreign_share"] = np.where(
        rst_totals["total_output_total"] != 0,
        100.0 * rst_totals["total_output_foreign"] / rst_totals["total_output_total"],
        np.nan,
    )

    # sector_type share in total regional output
    region_total_map = region_totals.set_index("iso")["total_output"]
    rst_totals["region_total_output"] = rst_totals["iso"].map(region_total_map)

    rst_totals["region_share"] = np.where(
        rst_totals["region_total_output"] != 0,
        100.0 * rst_totals["total_output_total"] / rst_totals["region_total_output"],
        np.nan,
    )

    rst_totals = (
        rst_totals
        .sort_values(["iso", "total_output_total"], ascending=[True, False])
        .reset_index(drop=True)
    )

    rst_totals = rst_totals[
        [
            "iso",
            "sector_type",
            "total_output_total",
            "total_output_domestic",
            "total_output_foreign",
            "domestic_share",
            "foreign_share",
            "region_share",
        ]
    ]

    return region_totals, rst_totals


def main():
    print("Loading MRIO file...")
    io_df = pd.read_pickle(MRIO_FILE)

    print("Loading sector table...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)

    print("Computing chunked output shares by region and sector type...")
    region_totals, rst_totals = compute_output_shares_by_sector_type_chunked(
        io_df,
        sector_table,
        chunk_size=CHUNK_SIZE
    )

    print(f"Saving {OUT_FILE_REGION}...")
    region_totals.to_csv(OUT_FILE_REGION, index=False)

    print(f"Saving {OUT_FILE_REGION_SECTOR_TYPE}...")
    rst_totals.to_csv(OUT_FILE_REGION_SECTOR_TYPE, index=False)

    print("Done!")


if __name__ == "__main__":
    main()