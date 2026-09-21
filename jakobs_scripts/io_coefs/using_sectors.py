#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
)

OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

TARGET_INPUT_SECTOR_NAME = "Vegetable products"
N_USING_SECTORS = 20


def main():
    print("=" * 70)
    print("USING-SECTOR ANALYSIS FOR GIVEN INPUT SECTOR")
    print("=" * 70)

    print(f"\nLoading MRIO from: {INPUT_PATH}")
    io_df = pd.read_pickle(INPUT_PATH)

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex (region, sector).")

    print(f"MRIO shape: {io_df.shape}")

    row_sectors = io_df.index.get_level_values(1)

    print(f"\nFinding rows for supplying sector: '{TARGET_INPUT_SECTOR_NAME}'")
    target_row_mask = row_sectors == TARGET_INPUT_SECTOR_NAME
    target_rows = io_df[target_row_mask]

    if target_rows.empty:
        raise ValueError(f"No rows found for sector '{TARGET_INPUT_SECTOR_NAME}'")

    print(
        f"Number of (region, '{TARGET_INPUT_SECTOR_NAME}') rows: "
        f"{target_rows.shape[0]}"
    )

    print(
        "\nComputing total use by each using sector (aggregated across using regions)..."
    )

    use_by_col = target_rows.sum(axis=0)  # (using_region, using_sector)

    use_by_using_sector = (
        use_by_col.groupby(use_by_col.index.get_level_values(1))
        .sum()
        .sort_values(ascending=False)
    )

    total_use = use_by_using_sector.sum()
    if total_use == 0:
        raise RuntimeError(
            f"Total use of sector '{TARGET_INPUT_SECTOR_NAME}' is zero; cannot compute shares."
        )

    shares = use_by_using_sector / total_use

    # ---- Ranking 1: Top by use of target input ----
    top_by_use = use_by_using_sector.head(N_USING_SECTORS)
    top_shares = shares.loc[top_by_use.index]

    top_by_use_table = pd.DataFrame(
        {
            "using_sector": top_by_use.index,
            "value": top_by_use.values,
            "pct_of_total_use_from_target": top_shares.values * 100.0,
        }
    )

    print(
        "\nTop using sectors by value of input from "
        f"'{TARGET_INPUT_SECTOR_NAME}' (aggregated across all regions):"
    )
    print(top_by_use_table.to_string(index=False))

    print("\nComputing total output by sector...")

    total_output_by_col = io_df.sum(axis=0)
    total_output_by_sector = (
        total_output_by_col.groupby(total_output_by_col.index.get_level_values(1))
        .sum()
        .sort_values(ascending=False)
    )

    using_sectors_with_use = use_by_using_sector[use_by_using_sector > 0].index
    total_output_restricted = total_output_by_sector.loc[
        total_output_by_sector.index.intersection(using_sectors_with_use)
    ]

    if total_output_restricted.empty:
        raise RuntimeError(
            "No sectors found that both have positive total output and "
            f"use input from '{TARGET_INPUT_SECTOR_NAME}'."
        )

    top_by_output = total_output_restricted.head(N_USING_SECTORS)

    print(
        "\nTop using sectors (by total output) among those that use input sector "
        f"'{TARGET_INPUT_SECTOR_NAME}' (aggregated across all regions):"
    )
    print(top_by_output)

    results_by_use = []
    for sec in top_by_use.index:
        share = float(top_shares.loc[sec])
        results_by_use.append(
            {
                "target_input_sector": TARGET_INPUT_SECTOR_NAME,
                "using_sector": sec,
                "ranking_type": "by_use",
                "total_output_value": float(total_output_by_sector.get(sec, 0.0)),
                "total_use_value_from_target": float(top_by_use.loc[sec]),
                "share_of_total_use_from_target": share,
                "pct_of_total_use_from_target": share * 100.0,
            }
        )
    df_by_use = pd.DataFrame(results_by_use)

    results_by_output = []
    for sec in top_by_output.index:
        share = float(shares.get(sec, 0.0))
        results_by_output.append(
            {
                "target_input_sector": TARGET_INPUT_SECTOR_NAME,
                "using_sector": sec,
                "ranking_type": "by_output",
                "total_output_value": float(top_by_output.loc[sec]),
                "total_use_value_from_target": float(use_by_using_sector.get(sec, 0.0)),
                "share_of_total_use_from_target": share,
                "pct_of_total_use_from_target": share * 100.0,
            }
        )
    df_by_output = pd.DataFrame(results_by_output)

    df_combined = pd.concat([df_by_use, df_by_output], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = TARGET_INPUT_SECTOR_NAME.replace(" ", "_").lower()

    out_use = OUT_DIR / f"{base}_top_using_sectors_by_use.csv"
    out_output = OUT_DIR / f"{base}_top_using_sectors_by_output.csv"
    out_combined = OUT_DIR / f"{base}_top_using_sectors_combined.csv"

    # df_by_use.to_csv(out_use, index=False)
    # df_by_output.to_csv(out_output, index=False)
    # df_combined.to_csv(out_combined, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
