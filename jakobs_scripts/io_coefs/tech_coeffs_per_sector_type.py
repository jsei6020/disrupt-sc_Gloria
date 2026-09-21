#!/usr/bin/env python3
import gc
from pathlib import Path
import pandas as pd
import numpy as np


# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_PATH = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_adjusted.pkl"
)
SECTOR_TABLE_FILE = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
)
OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

THRESHOLD = 0.001
CHUNK_SIZE = 1312
N_VA_ROWS = 164 * 6
N_FD_COLS = 164 * 6
MAX_SELF_COEF = 0.95


# ==============================================================================
# HELPERS
# ==============================================================================
def load_sector_mapping(sector_table_file: Path):
    print("\n[1/5] Loading sector table...")
    sector_table = pd.read_csv(sector_table_file)

    if "sector" not in sector_table.columns or "type" not in sector_table.columns:
        if "sector_name" in sector_table.columns:
            sector_table = sector_table.rename(columns={"sector_name": "sector"})
        if "sector_type" in sector_table.columns:
            sector_table = sector_table.rename(columns={"sector_type": "type"})

    if "sector" not in sector_table.columns or "type" not in sector_table.columns:
        raise ValueError(
            f"{sector_table_file} must contain columns "
            f"'sector' and 'type' (or aliases 'sector_name'/'sector_type')."
        )

    sector_name_to_type = sector_table.set_index("sector")["type"].to_dict()
    print(
        f"  Loaded mapping for {len(sector_name_to_type)} sector names "
        f"-> {sector_table['type'].nunique()} sector types"
    )
    return sector_name_to_type


def main():
    print("=" * 80)
    print("FULL MRIO TECHNICAL COEFFICIENT COUNT BY SUPPLIER SECTOR TYPE")
    print(f"Threshold: a_ij > {THRESHOLD}")
    print("=" * 80)

    print("\n[2/5] Loading MRIO...")
    io_df = pd.read_pickle(INPUT_PATH)
    n_rows, n_cols = io_df.shape
    print(f"  MRIO shape: {n_rows:,} x {n_cols:,}")

    # Production block only
    prod_rows = io_df.index[: n_rows - N_VA_ROWS]
    prod_cols = io_df.columns[: n_cols - N_FD_COLS]

    print(f"  Production rows: {len(prod_rows):,}")
    print(f"  Production cols: {len(prod_cols):,}")

    sector_name_to_type = load_sector_mapping(SECTOR_TABLE_FILE)

    # --------------------------------------------------------------------------
    # [3/5] Calculate buyer total outputs
    # a_ij = z_ij / x_j, where x_j is total output of buyer j
    # Following your original logic, x_j is the full row sum of each production
    # region-sector in the MRIO table.
    # --------------------------------------------------------------------------
    print("\n[3/5] Calculating total output vector (chunked)...")
    total_output_j = pd.Series(0.0, index=prod_cols, dtype="float32")

    for start in range(0, len(prod_cols), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(prod_cols))
        chunk = prod_cols[start:end]

        chunk_sum = io_df.loc[chunk].sum(axis=1).astype("float32")
        total_output_j.loc[chunk] = chunk_sum

        print(f"  Processed total output rows {start:,} to {end:,}")
        del chunk_sum
        gc.collect()

    valid_buyers = total_output_j[total_output_j > 0].index
    print(f"  Valid buyers with positive output: {len(valid_buyers):,}")

    # --------------------------------------------------------------------------
    # [4/5] For each supplier row, count coefficients above threshold
    # --------------------------------------------------------------------------
    print("\n[4/5] Computing coefficient counts row-wise (chunked)...")
    results = []

    # Pre-extract the production-to-production matrix only
    Z = io_df.loc[prod_rows, prod_cols]

    for start in range(0, len(prod_rows), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(prod_rows))
        row_chunk = prod_rows[start:end]

        # Chunk of supplier rows against all production buyer columns
        z_chunk = Z.loc[row_chunk, valid_buyers].astype("float32")

        # Divide each column by corresponding buyer total output
        coef_chunk = z_chunk.div(total_output_j.loc[valid_buyers], axis=1)

        # Cap self-coefficients for production sectors
        diag_keys = set(row_chunk).intersection(set(valid_buyers))
        capped_in_chunk = 0
        for key in diag_keys:
            val = coef_chunk.at[key, key]
            if pd.notna(val) and val > MAX_SELF_COEF:
                coef_chunk.at[key, key] = MAX_SELF_COEF
                capped_in_chunk += 1

        # Count coefficients above threshold for each supplier row
        counts = (coef_chunk > THRESHOLD).sum(axis=1).astype("int32")

        chunk_result = counts.rename("n_coeff_gt_threshold").reset_index()
        chunk_result.columns = ["supplier_region", "supplier_sector", "n_coeff_gt_threshold"]

        # Map sector -> type
        chunk_result["supplier_sector_type"] = chunk_result["supplier_sector"].map(
            sector_name_to_type
        )

        results.append(chunk_result)

        print(
            f"  Processed supplier rows {start:,} to {end:,} "
            f"(self-coefficients capped in chunk: {capped_in_chunk})"
        )

        del z_chunk, coef_chunk, counts, chunk_result
        gc.collect()

    supplier_counts = pd.concat(results, ignore_index=True)

    # Optional: keep unmapped sectors visible
    supplier_counts["supplier_sector_type"] = supplier_counts["supplier_sector_type"].fillna(
        "UNMAPPED"
    )

    # --------------------------------------------------------------------------
    # [5/5] Average by sector type and export
    # --------------------------------------------------------------------------
    print("\n[5/5] Aggregating by supplier sector type...")

    sector_type_summary = (
        supplier_counts.groupby("supplier_sector_type", dropna=False)
        .agg(
            avg_n_coeff_gt_threshold=("n_coeff_gt_threshold", "mean"),
            median_n_coeff_gt_threshold=("n_coeff_gt_threshold", "median"),
            min_n_coeff_gt_threshold=("n_coeff_gt_threshold", "min"),
            max_n_coeff_gt_threshold=("n_coeff_gt_threshold", "max"),
            n_supplier_rows=("n_coeff_gt_threshold", "size"),
        )
        .reset_index()
        .sort_values("avg_n_coeff_gt_threshold", ascending=False)
    )

    sector_type_summary["avg_n_coeff_gt_threshold"] = sector_type_summary[
        "avg_n_coeff_gt_threshold"
    ].astype("float32")
    sector_type_summary["median_n_coeff_gt_threshold"] = sector_type_summary[
        "median_n_coeff_gt_threshold"
    ].astype("float32")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detailed_out = OUT_DIR / f"supplier_row_counts_gt_{str(THRESHOLD).replace('.', 'p')}.csv"
    summary_out = OUT_DIR / f"avg_supplier_row_counts_by_sector_type_gt_{str(THRESHOLD).replace('.', 'p')}.csv"

    supplier_counts.to_csv(detailed_out, index=False)
    sector_type_summary.to_csv(summary_out, index=False)

    print(f"\n✅ DONE")
    print(f"Detailed row-level file: {detailed_out}")
    print(f"Sector-type summary file: {summary_out}")
    print(f"Total supplier rows: {len(supplier_counts):,}")
    print(f"Sector types: {sector_type_summary['supplier_sector_type'].nunique():,}")


if __name__ == "__main__":
    main()