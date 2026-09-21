#!/usr/bin/env python3
"""
Analyze main input sectors for Basic gold production.

Steps
-----
1. Load MRIO table from mrio_va_fd.pkl
2. Identify top 10 producing regions for sector 'Basic gold'
3. For these regions, compute:
   - Input mix by supplying sector (aggregated across all supplying regions)
   - Global aggregation of inputs over all 10 regions

Outputs
-------
1) basic_gold_top10_input_sectors_by_producer.csv
   Columns:
       basic_gold_region
       input_sector
       input_share_of_total_inputs
       input_value

2) basic_gold_top10_input_sectors_global.csv
   Columns:
       input_sector
       share_of_total_inputs
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
INPUT_PATH = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
)

OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

BASIC_GOLD_SECTOR_NAME = "Pulp and paper"
N_PRODUCERS = 10          # number of top producing regions to analyze
N_TOP_INPUTS_PER_REGION = 15   # how many largest inputs per producing region to store
N_TOP_INPUTS_GLOBAL = 20       # how many largest inputs in global aggregation

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BASIC GOLD INPUT-SECTOR ANALYSIS")
    print("=" * 70)

    # ----------------------
    # 1. Load MRIO
    # ----------------------
    print(f"\nLoading MRIO from: {INPUT_PATH}")
    io_df = pd.read_pickle(INPUT_PATH)

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex (region, sector).")

    print(f"MRIO shape: {io_df.shape}")

    row_regions = io_df.index.get_level_values(0)
    row_sectors = io_df.index.get_level_values(1)
    col_regions = io_df.columns.get_level_values(0)
    col_sectors = io_df.columns.get_level_values(1)

    # ----------------------
    # 2. Top 10 Basic gold producing regions
    # ----------------------
    print("\nFinding top Basic gold producers...")

    basic_gold_mask = row_sectors == BASIC_GOLD_SECTOR_NAME
    basic_gold_rows = io_df[basic_gold_mask]

    if basic_gold_rows.empty:
        raise ValueError(f"No rows found for sector '{BASIC_GOLD_SECTOR_NAME}'")

    # total output per producing region (sum over all columns)
    basic_gold_output = (
        basic_gold_rows
        .sum(axis=1)                           # index: (region, 'Basic gold')
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )

    top_regions = basic_gold_output.head(N_PRODUCERS).index.tolist()

    print("\nTop Basic gold producers:")
    print(basic_gold_output.head(N_PRODUCERS))

    # ----------------------
    # 3. Input analysis for each top region
    # ----------------------
    print("\nComputing input mixes for top Basic gold producers...")

    results = []

    for reg in top_regions:
        print(f"  Processing production region: {reg}")

        # columns where this region buys Basic gold
        col_mask = (col_regions == reg) & (col_sectors == BASIC_GOLD_SECTOR_NAME)

        if not col_mask.any():
            print(f"    WARNING: no columns found for ({reg}, {BASIC_GOLD_SECTOR_NAME}); skipping.")
            continue

        # all supplying rows into those Basic gold columns
        col_subset = io_df.loc[:, col_mask]          # rows: (sup_region, sup_sector)

        # total input by (supplying region, sector)
        input_by_row = col_subset.sum(axis=1)

        # aggregate across supplying regions -> global input by sector
        input_by_sector = (
            input_by_row
            .groupby(input_by_row.index.get_level_values(1))
            .sum()
            .sort_values(ascending=False)
        )

        total_inputs = input_by_sector.sum()
        if total_inputs == 0:
            print(f"    WARNING: zero total inputs for {reg}; skipping shares.")
            continue

        input_share = input_by_sector / total_inputs

        top_inputs = input_share.head(N_TOP_INPUTS_PER_REGION)

        for sec, share in top_inputs.items():
            results.append(
                {
                    "basic_gold_region": reg,
                    "input_sector": sec,
                    "input_share_of_total_inputs": float(share),
                    "input_value": float(input_by_sector.loc[sec]),
                }
            )

    if not results:
        raise RuntimeError("No input results computed; check sector names and MRIO structure.")

    inputs_df = pd.DataFrame(results)

    # ----------------------
    # 4. Global aggregation of inputs across all top regions
    # ----------------------
    print("\nAggregating inputs across all top Basic gold producers...")

    agg_inputs = (
        inputs_df
        .groupby("input_sector")["input_value"]
        .sum()
        .sort_values(ascending=False)
    )

    agg_total = agg_inputs.sum()
    if agg_total == 0:
        raise RuntimeError("Global total inputs are zero; cannot compute shares.")

    agg_shares = agg_inputs / agg_total
    top_global_inputs = agg_shares.head(N_TOP_INPUTS_GLOBAL)

    print("\nTop global input sectors for Basic gold (aggregated over top producers):")
    print(top_global_inputs)

    # ----------------------
    # 5. Export results
    # ----------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_producer_path = OUT_DIR / "basic_gold_top10_input_sectors_by_producer.csv"
    global_path = OUT_DIR / "basic_gold_top10_input_sectors_global.csv"

    inputs_df.to_csv(per_producer_path, index=False)

    top_global_inputs.to_frame("share_of_total_inputs").to_csv(
        global_path
    )

    print("\nResults written to:")
    print(f"  Per producer inputs: {per_producer_path}")
    print(f"  Global inputs:       {global_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
