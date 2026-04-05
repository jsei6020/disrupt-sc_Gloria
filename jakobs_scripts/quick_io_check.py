#!/usr/bin/env python3
import gc
from pathlib import Path
import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_PATH = Path(
    "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
)
OUT_DIR = Path(
    "/home/user/Documents/University/Master Thesis/supply_concentrations"
)

# Target sectors: compute their production-function coefficients
TARGET_PAIRS = [
    ("AUT", "Pulp and paper"),
    ("IDN", "Machinery and equipment"),
    ("JPN", "Basic nickel"),
]

# Threshold for reporting coefficients
COEFFICIENT_THRESHOLD = 0.0000

# Layout of MRIO: production block, then VA rows and FD cols
N_VA_ROWS = 164 * 6
N_FD_COLS = 164 * 6


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("TECHNICAL COEFFICIENTS FOR THREE TARGET SECTORS")
    print("Output-based and input-based, plus overall output/input ratio")
    print("=" * 70)

    print(f"\nLoading MRIO from: {INPUT_PATH}")
    io_df = pd.read_pickle(INPUT_PATH)
    print(f"Full MRIO shape: {io_df.shape}")

    n_rows, n_cols = io_df.shape

    # ------------------------------------------------------------------
    # 1) Define MRIO structure (as you described)
    # ------------------------------------------------------------------
    # Production block: top-left
    prod_rows = io_df.index[: n_rows - N_VA_ROWS]
    prod_cols = io_df.columns[: n_cols - N_FD_COLS]

    # VA rows at the bottom
    va_rows = io_df.index[n_rows - N_VA_ROWS :]

    # FD columns to the right
    fd_cols = io_df.columns[n_cols - N_FD_COLS :]

    print(f"Production rows: {len(prod_rows)}, production cols: {len(prod_cols)}")
    print(f"VA rows: {len(va_rows)}, FD cols: {len(fd_cols)}")

    # Core production block Z: all suppliers (rows) -> all production sectors (cols)
    Z = io_df.loc[prod_rows, prod_cols]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Loop over target sectors
    # ------------------------------------------------------------------
    for region, sector in TARGET_PAIRS:
        target = (region, sector)
        print(f"\n=== Target sector: {region} - {sector} ===")

        # Ensure target exists as both a row and a column in production block
        if target not in prod_cols:
            print(f"  ✗ Column not found for {target}, skipping.")
            continue
        if target not in prod_rows:
            print(f"  ✗ Row not found for {target}, skipping.")
            continue

        row_s = target
        col_s = target

        # ------------------------------------------------------------------
        # 2a) Total output_s:
        #      sum across the *row* s over production columns + FD columns
        # ------------------------------------------------------------------
        total_output_s = float(io_df.loc[row_s].sum())

        # ------------------------------------------------------------------
        # 2b) Total input_s:
        #      sum across the *column* s over production rows + VA rows
        # ------------------------------------------------------------------
        total_input_s = float(io_df.loc[:, col_s].sum())

        if total_output_s == 0 or total_input_s == 0:
            print(f"  ⚠ Zero total output or input for {target}, skipping.")
            continue

        overall_ratio = total_output_s / total_input_s

        # ------------------------------------------------------------------
        # 2c) Full production column for this target: all suppliers -> s
        # ------------------------------------------------------------------
        z_col = Z.loc[:, col_s]

        # Output-based coefficient for all suppliers i:
        #   tech_coeff_output_i = z_is / total_output_s
        tech_out_series = z_col / total_output_s

        # Input-based coefficient for all suppliers i:
        #   tech_coeff_input_i = z_is / total_input_s
        tech_in_series = z_col / total_input_s

        # Keep suppliers where either coefficient is above threshold
        mask_keep = (tech_out_series > COEFFICIENT_THRESHOLD) | (
            tech_in_series > COEFFICIENT_THRESHOLD
        )

        z_col = z_col[mask_keep]
        tech_out_series = tech_out_series[mask_keep]
        tech_in_series = tech_in_series[mask_keep]

        if z_col.empty:
            print(f"  ⚠ No coefficients above {COEFFICIENT_THRESHOLD} for {target}")
            continue

        # ------------------------------------------------------------------
        # 2d) Build output dataframe: one row per supplying region-sector
        # ------------------------------------------------------------------
        records = []
        for input_idx in z_col.index:
            sup_region, sup_sector = input_idx
            records.append(
                {
                    "input_region": sup_region,
                    "input_sector": sup_sector,
                    "tech_coeff_output": float(tech_out_series.loc[input_idx]),
                    "tech_coeff_input": float(tech_in_series.loc[input_idx]),
                    "overall_output_input_ratio": overall_ratio,
                }
            )

        df = pd.DataFrame(records)
        df.sort_values(
            by="tech_coeff_output",
            ascending=False,
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        out_name = (
            f"{region}_{sector.replace(' ', '_')}"
            f"_tech_coeffs_threshold_{COEFFICIENT_THRESHOLD}.csv"
        )
        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False)

        print(
            f"  ✓ Saved {len(df)} rows to {out_path} "
            f"(min coeff_out={df['tech_coeff_output'].min():.8f}, "
            f"max coeff_out={df['tech_coeff_output'].max():.8f})"
        )

        # Clean per-target intermediates
        del df, z_col, tech_out_series, tech_in_series
        gc.collect()

    # Cleanup big objects
    del io_df, Z
    gc.collect()

    print("\n✅ Done. One CSV per target sector with all supplying region_sectors.")


if __name__ == "__main__":
    main()
