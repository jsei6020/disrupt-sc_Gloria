#!/usr/bin/env python3
import gc
from pathlib import Path
import pandas as pd


# ==============================================================================
# CONFIG
# ==============================================================================

INPUT_PATH = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
)

OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

TARGET_PAIRS = [
    ("DEU", "Fabricated metal products"),
    ("IDN", "Fabricated metal products"),
    ("JPN", "Basic nickel"),
]

SUPPLIER = ("IDN", "Basic nickel")

N_VA_ROWS = 164 * 6
N_FD_COLS = 164 * 6
CHUNK_SIZE = 1312


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("COMPARISON: CHUNKED FULL-STYLE VS PER-SECTOR (3 TARGETS ONLY)")
    print("=" * 70)

    print(f"\nLoading MRIO from: {INPUT_PATH}")
    io_df = pd.read_pickle(INPUT_PATH)
    print(f"Full MRIO shape: {io_df.shape}")
    n_rows, n_cols = io_df.shape

    # Production block
    prod_rows = io_df.index[: n_rows - N_VA_ROWS]
    prod_cols = io_df.columns[: n_cols - N_FD_COLS]

    # Restrict to production block
    Z = io_df.loc[prod_rows, prod_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate targets
    valid_targets = []
    for t in TARGET_PAIRS:
        if t not in prod_rows:
            print(f"⚠ Target {t} missing as row in production block, skipping.")
            continue
        if t not in prod_cols:
            print(f"⚠ Target {t} missing as column in production block, skipping.")
            continue
        valid_targets.append(t)

    if not valid_targets:
        print("No valid targets found, exiting.")
        return

    # Check supplier exists as production row
    if SUPPLIER not in prod_rows:
        print(f"⚠ Supplier {SUPPLIER} not found as production row, exiting.")
        return

    print("hello")

    # ------------------------------------------------------------------
    # 1) Compute total output for ALL production rows, chunked
    #    -> but keep only a 1D Series, not a 2D matrix
    # ------------------------------------------------------------------
    chunk_size = 1000
    tot_outputs_list = []

    for i in range(0, len(prod_rows), chunk_size):
        rows_chunk = prod_rows[i : i + chunk_size]
        chunk_sum = io_df.loc[rows_chunk].sum(axis=1)  # sum over all columns
        tot_outputs_list.append(chunk_sum.astype("float32"))

    tot_outputs = pd.concat(tot_outputs_list)
    del tot_outputs_list
    gc.collect()

    # Ensure ordering/index matches prod_rows
    tot_outputs = tot_outputs.reindex(prod_rows).astype("float32")

    # ------------------------------------------------------------------
    # 2) Compute technical coefficients only for target columns, chunked
    # ------------------------------------------------------------------
    CHUNK_SIZE = 656  # tune to your memory
    results = []

    # Numerator block: all production rows, only target columns
    self_block = io_df.loc[prod_rows, valid_targets].astype("float32")

    for start in range(0, len(prod_rows), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(prod_rows))
        idx_slice = prod_rows[start:end]

        # Numerator: sub-block of Z for these rows and target columns
        num_chunk = self_block.loc[idx_slice, valid_targets]

        # Denominator: Series slice; Pandas will broadcast by index
        denom = tot_outputs.loc[idx_slice]

        # Division: each row i is divided by tot_outputs[i]
        chunk = num_chunk.div(denom, axis=0)

        results.append(chunk)
        del num_chunk, denom, chunk
        gc.collect()

    tech_coef_chunked = pd.concat(results, axis=0)
    del results, self_block, tot_outputs
    gc.collect()

    # Ensure row/column order
    tech_coef_chunked = tech_coef_chunked.loc[prod_rows, valid_targets]

    print(tech_coef_chunked)


    # ------------------------------------------------------------------
    # 3) Per-sector-style coefficients (second script), and comparison
    # ------------------------------------------------------------------
    records = []

    for target in valid_targets:
        region, sector = target
        print(f"\n=== Target: {region} - {sector} ===")

        # Per-sector total output (row sum across all columns)
        total_output_s = float(io_df.loc[target].sum())
        print(f"  Total output (per-sector logic): {total_output_s:.6e}")

        if total_output_s == 0:
            print(f"  ⚠ Zero total output for {target}, skipping.")
            continue

        # Column of Z: all production suppliers -> target
        z_col = Z.loc[:, target]

        # Per-sector output-based coefficients
        tech_out_series = z_col / total_output_s

        # Supplier-specific coefficients
        chunked_coeff = float(
            tech_coef_chunked.loc[SUPPLIER, target]
        ) if SUPPLIER in tech_coef_chunked.index else float("nan")

        per_sector_coeff = float(
            tech_out_series.loc[SUPPLIER]
        ) if SUPPLIER in tech_out_series.index else float("nan")

        records.append(
            {
                "target_region": region,
                "target_sector": sector,
                "supplier_region": SUPPLIER[0],
                "supplier_sector": SUPPLIER[1],
                "coeff_chunked_full_style": chunked_coeff,
                "coeff_per_sector": per_sector_coeff,
                "difference": chunked_coeff - per_sector_coeff,
            }
        )

    if not records:
        print("No records to write, exiting.")
        return

    df_out = pd.DataFrame(records)
    out_path = OUT_DIR / "IND_Basic_nickel_coeff_comparison_chunked_vs_per_sector.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n✓ Wrote comparison CSV to: {out_path}")
    print(df_out)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
