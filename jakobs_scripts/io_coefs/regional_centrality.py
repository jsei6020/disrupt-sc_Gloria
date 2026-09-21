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

THRESHOLD = 0.001
N_VA_ROWS = 164 * 6
N_FD_COLS = 164 * 6
MAX_SELF_COEF = 0.95


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 80)
    print("DOMESTIC HHI OF DEGREE CENTRALITY PER COUNTRY")
    print(f"Degree centrality = count of domestic technical coefficients > {THRESHOLD}")
    print("HHI computed on domestic degree shares only (diagonal country blocks)")
    print("=" * 80)

    print("\n[1/4] Loading MRIO...")
    io_df = pd.read_pickle(INPUT_PATH)
    n_rows, n_cols = io_df.shape
    print(f"  MRIO shape: {n_rows:,} x {n_cols:,}")

    # Production block only
    prod_rows = io_df.index[: n_rows - N_VA_ROWS]
    prod_cols = io_df.columns[: n_cols - N_FD_COLS]

    Z = io_df.loc[prod_rows, prod_cols].astype("float32")

    # Total output vector for buyers x_j
    print("\n[2/4] Calculating total output vector...")
    total_output_j = io_df.loc[prod_cols].sum(axis=1).astype("float32")
    total_output_j = total_output_j[total_output_j > 0]

    # Common region list from production rows and columns
    row_isos = pd.Index(prod_rows.get_level_values(0)).unique()
    col_isos = pd.Index(prod_cols.get_level_values(0)).unique()
    regions = sorted(set(row_isos).intersection(set(col_isos)))

    print(f"  Regions found: {len(regions)}")

    results = []

    print("\n[3/4] Processing domestic diagonal blocks...")
    for i, iso in enumerate(regions, start=1):
        print(f"  [{i}/{len(regions)}] {iso}")

        domestic_rows = prod_rows[prod_rows.get_level_values(0) == iso]
        domestic_cols = prod_cols[prod_cols.get_level_values(0) == iso]

        if len(domestic_rows) == 0 or len(domestic_cols) == 0:
            continue

        z_dom = Z.loc[domestic_rows, domestic_cols]

        valid_domestic_cols = total_output_j.index.intersection(domestic_cols)
        if len(valid_domestic_cols) == 0:
            results.append({
                "country_iso": iso,
                "n_supplier_sectors": int(len(domestic_rows)),
                "n_using_sectors": int(len(domestic_cols)),
                "n_valid_using_sectors": 0,
                "count_coeff_gt_threshold": 0,
                "total_domestic_degree": 0,
                "avg_degree_per_supplier_sector": 0.0,
                "max_degree": 0,
                "hhi_degree_domestic_0_1": np.nan,
                "hhi_degree_domestic_0_10000": np.nan,
            })
            continue

        z_dom = z_dom.loc[:, valid_domestic_cols]
        x_dom = total_output_j.loc[valid_domestic_cols]

        # Technical coefficients a_ij = z_ij / x_j
        a_dom = z_dom.div(x_dom, axis=1)

        # Optional regularization of self-coefficients
        common_diag = set(domestic_rows).intersection(set(valid_domestic_cols))
        for key in common_diag:
            val = a_dom.at[key, key]
            if pd.notna(val) and val > MAX_SELF_COEF:
                a_dom.at[key, key] = MAX_SELF_COEF

        # Binary domestic link matrix using threshold
        links = (a_dom > THRESHOLD)

        # Degree centrality per supplier sector:
        # number of domestic user sectors j connected to supplier sector i
        row_counts = links.sum(axis=1).astype("int32")

        total_degree = int(row_counts.sum())
        max_degree = int(row_counts.max()) if len(row_counts) > 0 else 0
        avg_degree = float(row_counts.mean()) if len(row_counts) > 0 else 0.0
        count_coeff_gt_threshold = int(links.to_numpy().sum())

        if total_degree > 0:
            degree_shares = row_counts / total_degree
            hhi_0_1 = float((degree_shares ** 2).sum())
            hhi_0_10000 = float(hhi_0_1 * 10000.0)
        else:
            hhi_0_1 = np.nan
            hhi_0_10000 = np.nan

        results.append({
            "country_iso": iso,
            "n_supplier_sectors": int(len(domestic_rows)),
            "n_using_sectors": int(len(domestic_cols)),
            "n_valid_using_sectors": int(len(valid_domestic_cols)),
            "count_coeff_gt_threshold": count_coeff_gt_threshold,
            "total_domestic_degree": total_degree,
            "avg_degree_per_supplier_sector": avg_degree,
            "max_degree": max_degree,
            "hhi_degree_domestic_0_1": hhi_0_1,
            "hhi_degree_domestic_0_10000": hhi_0_10000,
        })

        del domestic_rows, domestic_cols, valid_domestic_cols, z_dom, x_dom, a_dom, links, row_counts
        gc.collect()

    result_df = pd.DataFrame(results)
    result_df.sort_values("hhi_degree_domestic_0_10000", ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    print("\n[4/4] Exporting...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"domestic_hhi_degree_centrality_gt_{str(THRESHOLD).replace('.', 'p')}.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\n✅ DONE — File saved: {out_path}")
    print(f"Countries processed: {len(result_df)}")


if __name__ == "__main__":
    main()