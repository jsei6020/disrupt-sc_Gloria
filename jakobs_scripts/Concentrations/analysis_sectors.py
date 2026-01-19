#!/usr/bin/env python3
"""
Combined analysis script that:
1. Computes HHI of net-export shares for all sectors
2. For sectors with high concentration, identifies regions with >1% export share
3. Exports two CSV files with the results

CSV 1: region_sector_pairs_above_1pct.csv
    - Single column region_sector (e.g. USA_Water transport)
    - Columns: region_sector, supply_share_exports, total_supply_share, total_output, ~net export

CSV 2: sector_concentration_summary.csv
    - One row per sector with at least one exporter above threshold
    - Columns: sector, HHI_net_export_share, num_major_exporters,
               major_exporter_countries, combined_export_share
"""
import pandas as pd
import numpy as np


# ==== USER SETTINGS ====
INPUT_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
INPUT_FORMAT = "pkl"  # "pkl" or "parquet"

# Output paths
HHI_OUTPUT = "/home/user/Documents/University/Master Thesis/sector_hhi_net_export_share.csv"
REGION_SECTOR_PAIRS_OUTPUT = "/home/user/Documents/University/Master Thesis/supply_concentrations/region_sector_pairs_above_1pct.csv"
SECTOR_SUMMARY_OUTPUT = "/home/user/Documents/University/Master Thesis/supply_concentrations/sector_concentration_summary.csv"

# Threshold for export share (1%)
EXPORT_SHARE_THRESHOLD = 0.01

# Optional: Filter to analyze only sectors above a certain HHI threshold
# Set to None to analyze all sectors
HHI_THRESHOLD = 0.001  # e.g., 0.05 to only analyze sectors with HHI > 0.05
# ========================


def compute_hhi_net_export_by_sector(io_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector, compute the HHI of net-export shares.

    Assumes:
    - Rows: MultiIndex (region_iso3, sector_name)
    - Columns: MultiIndex (region_iso3, sector_name)

    Returns DataFrame with columns: sector, HHI_net_export_share
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    row_region_level = 0
    row_sector_level = 1
    col_region_level = 0

    sectors = io_df.index.get_level_values(row_sector_level).unique()
    results = []

    print(f"Computing HHI for {len(sectors)} sectors...")

    for i, sector_name in enumerate(sectors, 1):
        if i % 10 == 0:
            print(f"  Processing sector {i}/{len(sectors)}: {sector_name}")

        # Filter rows for this sector
        row_mask = io_df.index.get_level_values(row_sector_level) == sector_name
        sector_df = io_df[row_mask]

        if sector_df.empty:
            continue

        # Total supply per region (including domestic)
        row_sums = sector_df.sum(axis=1)
        region_totals = row_sums.groupby(
            row_sums.index.get_level_values(row_region_level)
        ).sum()
        world_total = region_totals.sum()

        # Exclude own-region columns to get exports
        col_regions = sector_df.columns.get_level_values(col_region_level)
        row_regions = sector_df.index.get_level_values(row_region_level)

        same_region_mask = pd.DataFrame(
            False, index=sector_df.index, columns=sector_df.columns
        )
        for reg in row_regions.unique():
            same_region_mask.loc[row_regions == reg, col_regions == reg] = True

        sector_excl_self = sector_df.mask(same_region_mask, 0.0)

        row_sums_excl_self = sector_excl_self.sum(axis=1)
        region_totals_excl_self = row_sums_excl_self.groupby(
            row_sums_excl_self.index.get_level_values(row_region_level)
        ).sum()

        # Net-export share per region
        net_export_share = region_totals_excl_self / world_total

        # HHI over these net-export shares
        hhi_net_export_share = (net_export_share ** 2).sum()

        results.append(
            {
                "sector": sector_name,
                "HHI_net_export_share": hhi_net_export_share,
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        "HHI_net_export_share", ascending=False
    ).reset_index(drop=True)
    return result_df


def compute_sector_output_shares(io_df: pd.DataFrame, sector_name: str) -> pd.DataFrame:
    """
    Compute regional output shares for a given sector from an IO table.

    Row and column index are expected to be MultiIndex: (region_iso3, sector_name).

    Returns DataFrame with columns:
        region, total_supply_share, total_output, ~net export, supply_share_exports
    """
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("Row index must be a MultiIndex with (region, sector).")
    if not isinstance(io_df.columns, pd.MultiIndex) or io_df.columns.nlevels < 2:
        raise ValueError("Column index must be a MultiIndex with (region, sector).")

    sector_level = 1
    row_region_level = 0
    col_region_level = 0

    # Filter rows for the specified sector
    row_mask = io_df.index.get_level_values(sector_level) == sector_name
    sector_df = io_df[row_mask]

    if sector_df.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "total_supply_share",
                "total_output",
                "~net export",
                "supply_share_exports",
            ]
        )

    # Totals including self
    row_sums = sector_df.sum(axis=1)
    region_totals = row_sums.groupby(
        row_sums.index.get_level_values(row_region_level)
    ).sum()
    world_total = region_totals.sum()
    total_supply_share = region_totals / world_total

    # Exclude own-region columns
    col_regions = sector_df.columns.get_level_values(col_region_level)
    row_regions = sector_df.index.get_level_values(row_region_level)

    same_region_mask = pd.DataFrame(
        False, index=sector_df.index, columns=sector_df.columns
    )
    for reg in row_regions.unique():
        same_region_mask.loc[row_regions == reg, col_regions == reg] = True

    sector_excl_self = sector_df.mask(same_region_mask, 0.0)

    row_sums_excl_self = sector_excl_self.sum(axis=1)
    region_totals_excl_self = row_sums_excl_self.groupby(
        row_sums_excl_self.index.get_level_values(row_region_level)
    ).sum()

    supply_share_exports = region_totals_excl_self / world_total

    # Build result
    result = pd.DataFrame(
        {
            "region": region_totals.index,
            "total_supply_share": total_supply_share.values,
            "total_output": region_totals.values,
            "~net export": region_totals_excl_self.reindex(
                region_totals.index
            ).values,
            "supply_share_exports": supply_share_exports.reindex(
                region_totals.index
            ).values,
        }
    )

    result = result.sort_values("supply_share_exports", ascending=False).reset_index(
        drop=True
    )
    return result


def main():
    print("=" * 60)
    print("COMBINED SECTOR CONCENTRATION ANALYSIS")
    print("=" * 60)

    # Load IO table
    print(f"\nLoading IO table from {INPUT_PATH}...")
    if INPUT_FORMAT == "pkl":
        io_df = pd.read_pickle(INPUT_PATH)
    elif INPUT_FORMAT == "parquet":
        io_df = pd.read_parquet(INPUT_PATH)
    else:
        raise ValueError("INPUT_FORMAT must be 'pkl' or 'parquet'.")

    print(f"Loaded IO table with shape: {io_df.shape}")

    # Step 1: Compute HHI for all sectors
    print("\n" + "=" * 60)
    print("STEP 1: Computing HHI for all sectors")
    print("=" * 60)
    hhi_df = compute_hhi_net_export_by_sector(io_df)
    hhi_df.to_csv(HHI_OUTPUT, index=False)
    print(f"\nHHI results saved to: {HHI_OUTPUT}")
    print(f"\nTop 10 sectors by HHI:")
    print(hhi_df.head(10).to_string(index=False))

    # Step 2: Filter sectors if threshold is set
    if HHI_THRESHOLD is not None:
        sectors_to_analyze = hhi_df[
            hhi_df["HHI_net_export_share"] >= HHI_THRESHOLD
        ]["sector"].tolist()
        print(
            f"\nAnalyzing {len(sectors_to_analyze)} sectors with HHI >= {HHI_THRESHOLD}"
        )
    else:
        sectors_to_analyze = hhi_df["sector"].tolist()
        print(f"\nAnalyzing all {len(sectors_to_analyze)} sectors")

    # Step 3: For each sector, find regions with export share > threshold
    print("\n" + "=" * 60)
    print(f"STEP 2: Finding regions with export share > {EXPORT_SHARE_THRESHOLD*100}%")
    print("=" * 60)

    all_region_sector_pairs = []
    sector_summaries = []

    for i, sector_name in enumerate(sectors_to_analyze, 1):
        if i % 10 == 0 or i == 1:
            print(f"Processing sector {i}/{len(sectors_to_analyze)}: {sector_name}")

        # Get output shares for this sector
        output_shares = compute_sector_output_shares(io_df, sector_name)

        # Filter regions above threshold
        significant_exporters = output_shares[
            output_shares["supply_share_exports"] >= EXPORT_SHARE_THRESHOLD
        ].copy()

        if len(significant_exporters) > 0:
            # Add sector name to each row
            significant_exporters["sector"] = sector_name

            # Add to list of all pairs
            all_region_sector_pairs.append(
                significant_exporters[
                    [
                        "sector",
                        "region",
                        "supply_share_exports",
                        "total_supply_share",
                        "total_output",
                        "~net export",
                    ]
                ]
            )

            # Create summary for this sector
            hhi_value = hhi_df[hhi_df["sector"] == sector_name][
                "HHI_net_export_share"
            ].values[0]
            countries_list = ", ".join(significant_exporters["region"].tolist())
            total_share = significant_exporters["supply_share_exports"].sum()

            sector_summaries.append(
                {
                    "sector": sector_name,
                    "HHI_net_export_share": hhi_value,
                    "num_major_exporters": len(significant_exporters),
                    "major_exporter_countries": countries_list,
                    "combined_export_share": total_share,
                }
            )

    # Step 4: Create and export CSV 1 - All region-sector pairs with combined key
    print("\n" + "=" * 60)
    print("STEP 3: Exporting results")
    print("=" * 60)

    if all_region_sector_pairs:
        region_sector_df = pd.concat(all_region_sector_pairs, ignore_index=True)

        # Create combined region_sector column (e.g. USA_Water transport) [web:9][web:18]
        region_sector_df["region_sector"] = (
            region_sector_df["region"].astype(str)
            + "_"
            + region_sector_df["sector"].astype(str)
        )

        # Keep region_sector first, then other numeric/info columns
        region_sector_df = region_sector_df[
            [
                "region_sector",
                "supply_share_exports",
                "total_supply_share",
                "total_output",
                "~net export",
            ]
        ]

        region_sector_df = region_sector_df.sort_values(
            "supply_share_exports", ascending=False
        ).reset_index(drop=True)

        region_sector_df.to_csv(REGION_SECTOR_PAIRS_OUTPUT, index=False)
        print(f"\nExported {len(region_sector_df)} region-sector pairs to:")
        print(f"  {REGION_SECTOR_PAIRS_OUTPUT}")
        print(f"\nTop 10 region-sector pairs by export share:")
        print(region_sector_df.head(10).to_string(index=False))
    else:
        print("\nNo region-sector pairs found above threshold.")

    # Step 5: Create and export CSV 2 - Sector summaries
    if sector_summaries:
        sector_summary_df = pd.DataFrame(sector_summaries)
        sector_summary_df = sector_summary_df.sort_values(
            "HHI_net_export_share", ascending=False
        ).reset_index(drop=True)

        sector_summary_df.to_csv(SECTOR_SUMMARY_OUTPUT, index=False)
        print(f"\nExported {len(sector_summary_df)} sector summaries to:")
        print(f"  {SECTOR_SUMMARY_OUTPUT}")
        print(f"\nTop 10 sectors by concentration:")
        print(sector_summary_df.head(10).to_string(index=False))
    else:
        print("\nNo sector summaries to export.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
