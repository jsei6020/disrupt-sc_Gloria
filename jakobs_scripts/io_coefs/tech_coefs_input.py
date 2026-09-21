#!/usr/bin/env python3
import gc
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
)
OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

TARGET_SECTOR_NAME = "Basic nickel"
TOP_N_COEFFICIENTS = 1000
CHUNK_SIZE = 1312          # row chunk size
N_VA_ROWS = 164 * 6        # number of value-added rows at the bottom
N_FD_COLS = 164 * 6


def main():
    print("=" * 70)
    print("IO COEFFICIENT ANALYSIS FOR SINGLE SECTOR (CHUNKED)")
    print("=" * 70)

    print(f"\nLoading MRIO from: {INPUT_PATH}")
    io_df: pd.DataFrame = pd.read_pickle(INPUT_PATH)

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex (region, sector).")

    print(f"MRIO shape: {io_df.shape}")

    # ------------------------------------------------------------------
    # 1. Intermediary part = all columns, but drop last N_VA_ROWS rows
    # ------------------------------------------------------------------
    if N_VA_ROWS >= io_df.shape[0]:
        raise ValueError("N_VA_ROWS >= number of rows in MRIO; check value-added row count.")

    intermediary_matrix = io_df.iloc[:-N_VA_ROWS, :-N_FD_COLS]
    print(f"Intermediary matrix shape: {intermediary_matrix.shape}")

    # ------------------------------------------------------------------
    # 2. Total outputs per region_sector (rows of intermediary matrix)
    # ------------------------------------------------------------------
    print("\nComputing total outputs per region_sector...")
    total_output = intermediary_matrix.sum(axis=1).astype("float32")

    # Drop zero-output rows from analysis
    nonzero_output_mask = total_output > 0
    valid_rows = total_output.index[nonzero_output_mask]
    total_output = total_output[nonzero_output_mask]

    print(f"Valid rows with non-zero output: {len(valid_rows)}")

    # ------------------------------------------------------------------
    # 3. Select all columns of TARGET_SECTOR_NAME (in intermediary matrix)
    # ------------------------------------------------------------------
    print(f"\nSelecting all uses of sector: {TARGET_SECTOR_NAME}")

    interm_col_sectors = intermediary_matrix.columns.get_level_values(1)
    target_col_mask = interm_col_sectors == TARGET_SECTOR_NAME

    if not target_col_mask.any():
        raise ValueError(f"No intermediary columns found for sector '{TARGET_SECTOR_NAME}'")

    target_cols = intermediary_matrix.columns[target_col_mask]

    # ------------------------------------------------------------------
    # 4. Chunk-wise computation of technical coefficients
    # ------------------------------------------------------------------
    print("\nComputing technical coefficients (chunked)...")

    rows_index = intermediary_matrix.index
    coeff_records = []

    for start in range(0, len(rows_index), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(rows_index))
        row_chunk_index = rows_index[start:end]

        row_chunk_index = [r for r in row_chunk_index if r in valid_rows]
        if not row_chunk_index:
            continue

        Z_chunk = intermediary_matrix.loc[row_chunk_index, target_cols].astype("float32")
        outputs_chunk = total_output.loc[row_chunk_index].astype("float32")

        denom_matrix = pd.DataFrame(
            np.repeat(outputs_chunk.to_numpy().reshape(-1, 1), Z_chunk.shape[1], axis=1),
            index=row_chunk_index,
            columns=target_cols,
        )

        A_chunk = Z_chunk / denom_matrix
        A_chunk = A_chunk.fillna(0.0)

        # A_chunk: index = (sup_region, sup_sector),
        #          columns = (supplied_region, supplied_sector)
        # We know supplied_sector is always TARGET_SECTOR_NAME for all columns.

        # 1) Move column MultiIndex into rows: stack only column level 0 (region)
        #    so we get a Series with index:
        #    (sup_region, sup_sector, supplied_region)
        stacked = A_chunk.stack(level=0)  # level 0 = supplied_region

        if stacked.empty:
            del Z_chunk, outputs_chunk, denom_matrix, A_chunk, stacked
            gc.collect()
            continue

        # 2) Turn into DataFrame and reset index
        long_chunk = stacked.reset_index()
        # columns are: ['sup_region', 'sup_sector', 'supplied_region', 0]
        long_chunk.columns = [
            "sup_region",
            "sup_sector",
            "supplied_region",
            "io_coefficient",
        ]

        # 3) Add supplied_sector (it is the same TARGET_SECTOR_NAME for all columns)
        long_chunk["supplied_sector"] = TARGET_SECTOR_NAME

        # 4) Reorder columns and filter positives
        long_chunk = long_chunk[
            ["sup_region", "sup_sector", "supplied_region", "supplied_sector", "io_coefficient"]
        ]
        long_chunk = long_chunk[long_chunk["io_coefficient"] > 0]

        coeff_records.append(long_chunk)

        del Z_chunk, outputs_chunk, denom_matrix, A_chunk, stacked, long_chunk
        gc.collect()


    if not coeff_records:
        raise RuntimeError("No positive IO coefficients computed. Check data and TARGET_SECTOR_NAME.")

    coeff_long = pd.concat(coeff_records, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Sort coefficients and compute average of top N
    # ------------------------------------------------------------------
    print("\nSorting coefficients and computing average of top values...")

    # total average over all positive coefficients
    total_avg_coeff = coeff_long["io_coefficient"].mean()

    # sort once (you already do this below)
    coeff_long.sort_values("io_coefficient", ascending=False, inplace=True)

    # average of top 10,000 coefficients
    top_n_for_avg = 10000
    top_10k = coeff_long.head(top_n_for_avg)
    top_10k_avg_coeff = top_10k["io_coefficient"].mean()

    print(f"Total average IO coefficient for sector '{TARGET_SECTOR_NAME}': {total_avg_coeff:.6f}")
    print(f"Average of top {top_n_for_avg} IO coefficients for sector '{TARGET_SECTOR_NAME}': {top_10k_avg_coeff:.6f}")

    top_coeffs = coeff_long.head(TOP_N_COEFFICIENTS).copy()
    avg_top_coeff = top_coeffs["io_coefficient"].mean()

    print(
        f"Average of top {TOP_N_COEFFICIENTS} IO coefficients "
        f"for sector '{TARGET_SECTOR_NAME}': {avg_top_coeff:.6f}"
    )

    # ------------------------------------------------------------------
    # 6. Export results
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_sector_name = TARGET_SECTOR_NAME.replace(" ", "_")
    top_coeffs_path = (
        OUT_DIR
        / f"io_coefficients_top_{TOP_N_COEFFICIENTS}_{safe_sector_name}.csv"
    )

    top_coeffs.to_csv(top_coeffs_path, index=False)

    print("\nResults written to:")
    print(f"  Top coefficients: {top_coeffs_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
