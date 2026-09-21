import gc
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_all.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
FD_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/region_sector_type_final_demand_with_shares.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"

OUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-is-st_fd.csv"
SECTOR_AVG_OUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-astis_fd.csv"


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: REGION-SECTOR-TYPE LOSS INTENSITY (CLIENT SIDE)")
    print("=" * 80)

    # --- 1. Load Runs Mapping ---
    print("\n[1/6] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]
    print(f"  Found {len(runs)} valid simulations")

    # --- 2. Load Region Name Mapping ---
    print("\n[2/6] Loading region table...")
    try:
        regions_df = pd.read_csv(REGIONS_TABLE_FILE)
        if "Region_acronyms" not in regions_df.columns or "Region_names" not in regions_df.columns:
            raise ValueError("Expected 'Region_acronyms' and 'Region_names' columns in regions_table.csv")

        region_name_map = dict(
            zip(
                regions_df["Region_acronyms"].astype(str),
                regions_df["Region_names"].astype(str)
            )
        )
        print(f"  Loaded region names for {len(region_name_map)} regions")
    except Exception as e:
        print(f"  ERROR loading {REGIONS_TABLE_FILE}: {e}")
        return

    # --- 3. Load Sector Mapping ---
    print("\n[3/6] Loading sector table...")
    try:
        sector_table = pd.read_csv(SECTOR_TABLE_FILE)
        if 'sector' not in sector_table.columns or 'type' not in sector_table.columns:
            if 'sector_name' in sector_table.columns:
                sector_table.rename(columns={'sector_name': 'sector'}, inplace=True)
            if 'sector_type' in sector_table.columns:
                sector_table.rename(columns={'sector_type': 'type'}, inplace=True)

        sector_name_to_type = sector_table.set_index('sector')['type'].to_dict()
        print(f"  Loaded mapping for {len(sector_name_to_type)} sector names")
    except Exception as e:
        print(f"  ERROR loading {SECTOR_TABLE_FILE}: {e}")
        return

    # --- 4. Load Economic Output Data ---
    print("\n[4/6] Loading region-sector-type output data...")
    try:
        df_output = pd.read_csv(FD_FILE)
        if df_output.columns[0] != 'iso':
            df_output.rename(columns={df_output.columns[0]: 'iso'}, inplace=True)
        if df_output.columns[1] != 'sector_type':
            df_output.rename(columns={df_output.columns[1]: 'sector_type'}, inplace=True)
        if df_output.columns[2] != 'final_demand_total':
            df_output.rename(columns={df_output.columns[2]: 'final_demand_total'}, inplace=True)

        output_map = {}
        for _, row in df_output.iterrows():
            output_map[(row['iso'], row['sector_type'])] = row['final_demand_total']

        print(f"  Loaded economic data for {len(output_map)} region-sector-type combinations")
    except Exception as e:
        print(f"  ERROR loading {OUTPUT_FILE}: {e}")
        return

    # --- 5. Accumulate Losses ---
    print("\n[5/6] Processing loss files...")
    region_sector_type_loss_sum: Dict[Tuple[str, str], float] = {}
    processed_count = 0

    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")

            try:
                df_loss = pd.read_csv(loss_path)

                if 'region' not in df_loss.columns or 'sector' not in df_loss.columns or 'loss' not in df_loss.columns:
                    del df_loss
                    continue

                df_loss['sector_name'] = df_loss['sector'].str.split('_', n=1).str[1]
                df_loss['sector_type'] = df_loss['sector_name'].map(sector_name_to_type)
                df_loss = df_loss.dropna(subset=['sector_type'])

                if len(df_loss) == 0:
                    del df_loss
                    continue

                file_agg = df_loss.groupby(['region', 'sector_type'])['loss'].sum()

                for key, loss_val in file_agg.items():
                    region_sector_type_loss_sum[key] = region_sector_type_loss_sum.get(key, 0.0) + loss_val

                del df_loss
                del file_agg
                gc.collect()
                processed_count += 1

            except Exception as e:
                print(f"\n  ERROR processing {folder}: {e}")
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")

    print(f"\n  Completed processing {processed_count} simulation files")

    if processed_count <= 0:
        print("\nERROR: No simulation files were processed successfully.")
        return

    # --- 6. Calculate Intensity & Export ---
    print("\n[6/6] Calculating intensity scores and saving...")
    results = []

    for key, total_loss in region_sector_type_loss_sum.items():
        iso, sector_type = key
        final_demand_total = output_map.get(key, None)

        if final_demand_total is None:
            continue
        if final_demand_total <= 0:
            continue

        intensity_score = total_loss / (final_demand_total * processed_count)

        results.append({
            'iso': iso,
            'region_name': region_name_map.get(iso, iso),
            'sector_type': sector_type,
            'total_loss': total_loss,
            'final_demand_total': final_demand_total,
            'intensity_score': intensity_score
        })

    df_results = pd.DataFrame(results)

    if len(df_results) > 0:
        df_results = df_results.sort_values('intensity_score', ascending=False).reset_index(drop=True)
        df_results.to_csv(OUT_FILE, index=False)

        print(f"\n{'=' * 80}")
        print(f"DONE! Main results saved to {OUT_FILE}")
        print(f"{'=' * 80}")

        print("\nTOP 10 MOST VULNERABLE REGION-SECTOR-TYPES (BY INTENSITY):")
        print(df_results[['iso', 'region_name', 'sector_type', 'intensity_score', 'total_loss']].head(10).to_string(index=True))

        df_sector_avg = (
            df_results
            .groupby('sector_type', as_index=False)['intensity_score']
            .mean()
            .rename(columns={'intensity_score': 'avg_intensity_across_countries'})
            .sort_values('avg_intensity_across_countries', ascending=False)
            .reset_index(drop=True)
        )

        df_sector_avg.to_csv(SECTOR_AVG_OUT_FILE, index=False)

        print(f"\nSector-average results saved to {SECTOR_AVG_OUT_FILE}")
        print("\nTOP 10 SECTOR TYPES BY AVERAGE INTENSITY ACROSS COUNTRIES:")
        print(df_sector_avg.head(10).to_string(index=True))
    else:
        print("\nERROR: No results generated. Check sector mapping and region matching between loss files and output file.")


if __name__ == "__main__":
    main()