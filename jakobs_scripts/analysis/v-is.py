import gc
import pandas as pd
from pathlib import Path
from typing import Dict


# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final12weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status_threat.csv"
OUTPUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/total_region_output.csv"
REGIONS_TABLE_FILE = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"

OUT_FILE = "/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-is_threat.csv"


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: REGIONAL LOSS INTENSITY")
    print("=" * 80)

    # --- 1. Load Runs Mapping ---
    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    runs = runs[
        (runs["exists_in_global4_final"] == True) &
        (runs["region_sector"] != "UNKNOWN (extra folder)")
    ].copy()
    print(f"  Found {len(runs)} valid simulations")

    # --- 2. Load Region Name Mapping ---
    print("\n[2/5] Loading region table...")
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

    # --- 3. Load Economic Output Data ---
    print("\n[3/5] Loading total region output data...")
    try:
        df_output = pd.read_csv(OUTPUT_FILE)
        if df_output.columns[0] != 'iso':
            df_output.rename(columns={df_output.columns[0]: 'iso'}, inplace=True)
        if df_output.columns[1] != 'total_output':
            df_output.rename(columns={df_output.columns[1]: 'total_output'}, inplace=True)

        output_map = df_output.set_index('iso')['total_output'].to_dict()
        print(f"  Loaded economic data for {len(output_map)} regions")
    except Exception as e:
        print(f"  ERROR loading {OUTPUT_FILE}: {e}")
        return

    # --- 4. Accumulate Losses per Region ---
    print("\n[4/5] Processing loss files...")
    region_loss_sum: Dict[str, float] = {}
    processed_count = 0

    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"

        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")

            try:
                df_loss = pd.read_csv(loss_path)

                if 'region' not in df_loss.columns or 'loss' not in df_loss.columns:
                    print(f"\n  WARNING: {folder} missing required columns. Skipping.")
                    del df_loss
                    continue

                file_agg = df_loss.groupby('region')['loss'].sum()

                for region_iso, loss_val in file_agg.items():
                    region_loss_sum[region_iso] = region_loss_sum.get(region_iso, 0.0) + loss_val

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

    # --- 5. Calculate Intensity Score & Export ---
    print("\n[5/5] Calculating intensity scores and saving...")
    results = []

    for region_iso, total_loss in region_loss_sum.items():
        total_output = output_map.get(region_iso, None)

        if total_output is None:
            continue
        if total_output <= 0:
            continue

        intensity_score = total_loss / (total_output * processed_count)

        results.append({
            'iso': region_iso,
            'region_name': region_name_map.get(region_iso, region_iso),
            'total_loss': total_loss,
            'total_output': total_output,
            'intensity_score': intensity_score
        })

    df_results = pd.DataFrame(results)

    if len(df_results) > 0:
        df_results = df_results.sort_values('intensity_score', ascending=False).reset_index(drop=True)
        df_results.to_csv(OUT_FILE, index=False)

        print(f"\n{'=' * 80}")
        print(f"DONE! Results saved to {OUT_FILE}")
        print(f"{'=' * 80}")

        print("\nTOP 10 MOST VULNERABLE REGIONS (BY INTENSITY):")
        print(df_results[['iso', 'region_name', 'intensity_score', 'total_loss']].head(10).to_string(index=True))
    else:
        print("\nERROR: No results generated. Check region matching between loss files and output file.")


if __name__ == "__main__":
    main()