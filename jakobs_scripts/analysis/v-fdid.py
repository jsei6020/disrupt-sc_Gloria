#!/usr/bin/env python3
"""
Compute two import dependency measures for each region in the MRIO:

1. overall_import_dependency:
   imports into all columns of region r / total use of all columns of region r

2. final_demand_import_dependency:
   imported final demand / total final demand
   using the 6 final demand categories listed in fd_names.csv

This version includes value added rows if present.
"""

import pandas as pd
import numpy as np

# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_adjusted.pkl"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"
INPUT_FORMAT = "pkl"  # "pkl" or "csv"

OUTPUT_PATH = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-fdid.csv"

FINAL_DEMAND_LABELS = {
    "Household final consumption P.3h",
    "Non-profit institutions serving households P.3n",
    "Government final consumption P.3g",
    "Gross fixed capital formation P.51",
    "Changes in inventories P.52",
    "Acquisitions less disposals of valuables P.53",
}
# =======================


def main():
    print("=" * 60)
    print("IMPORT DEPENDENCY PER REGION")
    print("=" * 60)

    print(f"\nLoading region table from {REGIONS_TABLE_FILE}...")
    regions_df = pd.read_csv(REGIONS_TABLE_FILE)

    if "Region_acronyms" in regions_df.columns:
        region_list = regions_df["Region_acronyms"].astype(str).tolist()
    else:
        raise ValueError("Expected a 'Region_acronyms' column in regions_table.csv")

    print(f"Number of regions in region table: {len(region_list)}")

    print(f"\nLoading IO table from {INPUT_PATH}...")
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "csv":
        io_df = pd.read_csv(INPUT_PATH, header=[0, 1], index_col=[0, 1])
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'csv'.")

    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, item).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, item).")

    print(f"Loaded IO table with shape: {io_df.shape}")

    row_regions = io_df.index.get_level_values(0).astype(str)
    col_regions = io_df.columns.get_level_values(0).astype(str)
    col_items = io_df.columns.get_level_values(1).astype(str)

    regions_to_process = [r for r in region_list if r in set(col_regions)]
    print(f"Regions found in MRIO columns: {len(regions_to_process)}")

    missing_col_regions = [r for r in region_list if r not in set(col_regions)]
    if missing_col_regions:
        print("Regions in region table but not in MRIO columns:")
        print(missing_col_regions)

    if "Region_acronyms" in regions_df.columns and "Region_names" in regions_df.columns:
        region_list = regions_df["Region_acronyms"].astype(str).tolist()
        region_name_map = dict(
            zip(
                regions_df["Region_acronyms"].astype(str),
                regions_df["Region_names"].astype(str)
            )
        )
    else:
        raise ValueError("Expected 'Region_acronyms' and 'Region_names' columns in regions_table.csv")

    found_fd_labels = sorted(set(col_items).intersection(FINAL_DEMAND_LABELS))
    missing_fd_labels = sorted(FINAL_DEMAND_LABELS.difference(set(col_items)))

    print("\nFinal demand labels found in MRIO:")
    print(found_fd_labels)

    if missing_fd_labels:
        print("\nWARNING: These final demand labels were not found:")
        print(missing_fd_labels)

    results = []

    print("\n" + "=" * 60)
    print("STEP: Computing import dependency metrics")
    print("=" * 60)

    for i, reg in enumerate(regions_to_process, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Processing region {i}/{len(regions_to_process)}: {reg}")
        import_rows_mask = (row_regions != reg)
        # 1) Overall import dependency
        all_region_cols_mask = (col_regions == reg)
        all_region_block = io_df.loc[:, all_region_cols_mask]
        total_consumption = float(np.nansum(all_region_block.to_numpy()))
        imports_total = float(np.nansum(all_region_block.loc[import_rows_mask, :].to_numpy()))

        overall_import_dependency = (
            imports_total / total_consumption if total_consumption != 0 else np.nan
        )

        # 2) Final-demand import dependency
        fd_cols_mask = (col_regions == reg) & (col_items.isin(FINAL_DEMAND_LABELS))
        fd_block = io_df.loc[:, fd_cols_mask]

        total_final_demand = float(np.nansum(fd_block.to_numpy()))
        imported_final_demand = float(np.nansum(fd_block.loc[import_rows_mask, :].to_numpy()))

        final_demand_import_dependency = (
            imported_final_demand / total_final_demand
            if total_final_demand != 0
            else np.nan
        )

        results.append(
            {
                "iso": reg,
                "region_name": region_name_map.get(reg, reg),
                "total_consumption": total_consumption,
                "imports_total": imports_total,
                "overall_import_dependency": overall_import_dependency,
                "total_final_demand": total_final_demand,
                "imported_final_demand": imported_final_demand,
                "final_demand_import_dependency": final_demand_import_dependency,
            }
        )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="final_demand_import_dependency",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    result_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")
    print("\nSample output:")
    print(result_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()