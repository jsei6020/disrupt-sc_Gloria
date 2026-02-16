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
N_VA_ROWS = 164 * 6        # value-added rows at the bottom
N_FD_COLS = 164 * 6        # final-demand columns at the right


def main():
    print("=" * 70)
    print("IO COEFFICIENT ANALYSIS FOR SINGLE SECTOR AS SUPPLIER (CHUNKED)")
    print("=" * 70)

    print(f"\nLoading MRIO from: {INPUT_PATH}")
    io_df: pd.DataFrame = pd.read_pickle(INPUT_PATH)

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex (region, sector).")

    print(f"MRIO shape: {io_df.shape}")

    n_rows, n_cols = io_df.shape
    if N_VA_ROWS >= n_rows or N_FD_COLS >= n_cols:
        raise ValueError("N_VA_ROWS or N_FD_COLS too large for MRIO dimensions.")

    # 1. Intermediary block: drop VA rows and FD columns
    intermediary_matrix = io_df.iloc[:-N_VA_ROWS, :-N_FD_COLS]
    print(f"Intermediary matrix shape: {intermediary_matrix.shape}")

    # 2. Select all ROWS of TARGET_SECTOR_NAME (suppliers)
    print(f"\nSelecting all supplying rows of sector: {TARGET_SECTOR_NAME}")
    interm_row_sectors = intermediary_matrix.index.get_level_values(1)
    target_row_mask = interm_row_sectors == TARGET_SECTOR_NAME

    if not target_row_mask.any():
        raise ValueError(f"No intermediary rows found for sector '{TARGET_SECTOR_NAME}'")

    target_rows = intermediary_matrix.index[target_row_mask]
    print(f"Number of supplier rows (regions) for {TARGET_SECTOR_NAME}: {len(target_rows)}")

    # 3. Total outputs per BUYING region-sector (columns): a_ij = z_ij / x_j
    print("\nComputing total outputs per buying region_sector (columns)...")
    total_output_cols = intermediary_matrix.sum(axis=0).astype("float32")
    nonzero_cols = total_output_cols[total_output_cols > 0]
    valid_cols = nonzero_cols.index
    print(f"Valid buying columns with non-zero output: {len(valid_cols)}")

    # 4. Chunk-wise computation of technical coefficients + melt
    print("\nComputing technical coefficients (chunked)...")
    coeff_records = []

    for start in range(0, len(target_rows), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(target_rows))
        row_chunk_index = target_rows[start:end]

        # Flows from TARGET_SECTOR_NAME rows to all valid buyer columns
        Z_chunk = intermediary_matrix.loc[row_chunk_index, valid_cols].astype("float32")

        # Buyer outputs (same for all rows in this chunk)
        outputs_chunk = total_output_cols.loc[valid_cols].astype("float32")

        # Denominator: repeat column outputs across rows
        denom_matrix = pd.DataFrame(
            np.repeat(outputs_chunk.to_numpy().reshape(1, -1), Z_chunk.shape[0], axis=0),
            index=row_chunk_index,
            columns=valid_cols,
        )

        A_chunk = Z_chunk / denom_matrix
        A_chunk = A_chunk.fillna(0.0)

        # Wide -> long via reset_index + melt (no stack)
        # Reset index so row MultiIndex becomes columns
        A_chunk = A_chunk.reset_index()
        # Name those two columns explicitly
        A_chunk.columns = ["sup_region", "sup_sector"] + list(A_chunk.columns[2:])

        id_vars = ["sup_region", "sup_sector"]

        long_chunk = A_chunk.melt(
            id_vars=id_vars,
            var_name="supplied_region_sector",
            value_name="io_coefficient",
        )

        # supplied_region_sector is a tuple (region, sector); split into two columns
        supplied_df = pd.DataFrame(
            long_chunk["supplied_region_sector"].tolist(),
            index=long_chunk.index,
            columns=["supplied_region", "supplied_sector"],
        )
        long_chunk = pd.concat([long_chunk.drop(columns=["supplied_region_sector"]), supplied_df], axis=1)

        long_chunk = long_chunk[
            ["sup_region", "sup_sector", "supplied_region", "supplied_sector", "io_coefficient"]
        ]
        long_chunk = long_chunk[long_chunk["io_coefficient"] > 0]

        coeff_records.append(long_chunk)

        del Z_chunk, outputs_chunk, denom_matrix, A_chunk, long_chunk, supplied_df
        gc.collect()

    if not coeff_records:
        raise RuntimeError("No positive IO coefficients computed. Check data and TARGET_SECTOR_NAME.")

    coeff_long = pd.concat(coeff_records, ignore_index=True)

    # 5. Averages (total, top 10k, top 1k) and export top 1k
    print("\nSorting coefficients and computing averages...")

    total_avg_coeff = coeff_long["io_coefficient"].mean()

    coeff_long.sort_values("io_coefficient", ascending=False, inplace=True)

    top_n_for_avg = 10000
    top_10k = coeff_long.head(top_n_for_avg)
    top_10k_avg_coeff = top_10k["io_coefficient"].mean()

    print(f"Total average IO coefficient for sector '{TARGET_SECTOR_NAME}' as supplier: {total_avg_coeff:.6f}")
    print(f"Average of top {top_n_for_avg} IO coefficients for sector '{TARGET_SECTOR_NAME}' as supplier: {top_10k_avg_coeff:.6f}")

    top_coeffs = coeff_long.head(TOP_N_COEFFICIENTS).copy()
    avg_top_coeff = top_coeffs["io_coefficient"].mean()

    print(
        f"Average of top {TOP_N_COEFFICIENTS} IO coefficients "
        f"for sector '{TARGET_SECTOR_NAME}' as supplier: {avg_top_coeff:.6f}"
    )

    # 6. Export results
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_sector_name = TARGET_SECTOR_NAME.replace(" ", "_")

    top_coeffs_path = (
        OUT_DIR
        / f"io_coefficients_top_{TOP_N_COEFFICIENTS}_{safe_sector_name}_as_supplier.csv"
    )
    averages_path = (
        OUT_DIR
        / f"io_coefficients_averages_{safe_sector_name}_as_supplier.csv"
    )

    top_coeffs.to_csv(top_coeffs_path, index=False)

    pd.DataFrame(
        [
            {
                "target_sector": TARGET_SECTOR_NAME,
                "metric": "total_average",
                "top_n": coeff_long.shape[0],
                "io_coefficient": total_avg_coeff,
            },
            {
                "target_sector": TARGET_SECTOR_NAME,
                "metric": "top_1000_average",
                "top_n": TOP_N_COEFFICIENTS,
                "io_coefficient": avg_top_coeff,
            },
            {
                "target_sector": TARGET_SECTOR_NAME,
                "metric": "top_10000_average",
                "top_n": top_n_for_avg,
                "io_coefficient": top_10k_avg_coeff,
            },
        ]
    ).to_csv(averages_path, index=False)

    print("\nResults written to:")
    print(f"  Top coefficients: {top_coeffs_path}")
    print(f"  Averages summary: {averages_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
