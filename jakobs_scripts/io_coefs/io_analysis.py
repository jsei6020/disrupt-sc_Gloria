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
OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

SUPPLIER_REGION = "IDN"
SUPPLIER_SECTOR = "Basic nickel"
COEFFICIENT_CUTOFF = 0.0001
CHUNK_SIZE = 1312
N_VA_ROWS = 164 * 6
N_FD_COLS = 164 * 6

# ==============================================================================
# MAIN — 100% CHUNKED, VECTORIZED, PRE-ALLOCATED, NO LOOPS
# ==============================================================================
def main():
    print("=" * 80)
    print("FINAL: FULLY CHUNKED VECTORIZED — NO PER-ROW LOOPS")
    print("a_ij = z_ij / buyer_total_output (full row sum, chunked batch sum)")
    print("=" * 80)

    io_df = pd.read_pickle(INPUT_PATH)
    n_rows, n_cols = io_df.shape

    # Production rows & columns (intermediate sectors)
    prod_rows = io_df.index[: n_rows - N_VA_ROWS]
    prod_cols = io_df.columns[: n_cols - N_FD_COLS]

    # --------------------------------------------------------------------------
    # PRE-ALLOCATE total_output SERIES (PREPARED VECTOR)
    # --------------------------------------------------------------------------
    total_output_j = pd.Series(0.0, index=prod_cols, dtype="float32")

    # --------------------------------------------------------------------------
    # CHUNKED + VECTORIZED total output calculation (NO PER-ROW LOOPS)
    # Directly assign chunk sums into PREPARED total_output vector
    # --------------------------------------------------------------------------
    print("\nCalculating total output (chunked vectorized)...")
    for start in range(0, len(prod_cols), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(prod_cols))
        chunk = prod_cols[start:end]

        # Vectorized sum for entire chunk — NO LOOPS
        chunk_sum = io_df.loc[chunk].sum(axis=1).astype("float32")
        
        # APPEND DIRECTLY TO PREPARED total_output VECTOR
        total_output_j.loc[chunk] = chunk_sum

        del chunk_sum
        gc.collect()

    valid_buyers = total_output_j[total_output_j > 0].index

    # --------------------------------------------------------------------------
    # Get supplier row
    # --------------------------------------------------------------------------
    supplier = (SUPPLIER_REGION, SUPPLIER_SECTOR)
    supplier_flows = io_df.loc[supplier, prod_cols].astype("float32")

    # --------------------------------------------------------------------------
    # CHUNKED COEFFICIENTS
    # --------------------------------------------------------------------------
    results = []
    for start in range(0, len(valid_buyers), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(valid_buyers))
        chunk = valid_buyers[start:end]

        z = supplier_flows.loc[chunk]
        out = total_output_j.loc[chunk]
        coeff = z / out
        coeff = coeff.where(coeff >= COEFFICIENT_CUTOFF).dropna()

        df = coeff.rename("io_coefficient").reset_index()
        df.columns = ["buyer_region", "buyer_sector", "io_coefficient"]
        df.insert(0, "supplier_sector", SUPPLIER_SECTOR)
        df.insert(0, "supplier_region", SUPPLIER_REGION)
        results.append(df)

        del z, out, coeff, df
        gc.collect()

    # --------------------------------------------------------------------------
    # Final sort & export
    # --------------------------------------------------------------------------
    final = pd.concat(results, ignore_index=True)
    final.sort_values("io_coefficient", ascending=False, inplace=True)
    final.reset_index(drop=True, inplace=True)

    OUT_DIR.mkdir(exist_ok=True)
    safe_r = SUPPLIER_REGION.replace(" ", "_")
    safe_s = SUPPLIER_SECTOR.replace(" ", "_")
    out_path = OUT_DIR / f"coeffs_{safe_r}_{safe_s}_descending.csv"
    final.to_csv(out_path, index=False)

    print(f"\n✅ DONE — File saved: {out_path}")
    print(f"Total coefficients: {len(final)}")

if __name__ == "__main__":
    main()