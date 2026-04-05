import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final8weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status2.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
OUTPUT_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/region_sector_type_total_output.csv"
OUT_FILE = "vulnerability_intensity_rankings_sector_type.csv"

def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: REGION-SECTOR-TYPE LOSS INTENSITY (CLIENT SIDE)")
    print("=" * 80)
    
    # --- 1. Load Runs Mapping ---
    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]
    print(f"  Found {len(runs)} valid simulations")

    # --- 2. Load Sector Mapping (Name -> Type) ---
    print("\n[2/5] Loading sector table (for name-to-type mapping)...")
    try:
        sector_table = pd.read_csv(SECTOR_TABLE_FILE)
        # Ensure correct column names
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

    # --- 3. Load Economic Output Data (ISO + Sector Type -> Output) ---
    print("\n[3/5] Loading region-sector-type output data...")
    try:
        df_output = pd.read_csv(OUTPUT_FILE)
        # Ensure standard column names
        if df_output.columns[0] != 'iso':
            df_output.rename(columns={df_output.columns[0]: 'iso'}, inplace=True)
        if df_output.columns[1] != 'sector_type':
            df_output.rename(columns={df_output.columns[1]: 'sector_type'}, inplace=True)
        if df_output.columns[2] != 'total_output':
            df_output.rename(columns={df_output.columns[2]: 'total_output'}, inplace=True)
            
        # Create a map keyed by (iso, sector_type) tuple for fast lookup
        # Note: 'iso' here refers to the AFFECTED region (from loss file 'region' column)
        output_map = {}
        for _, row in df_output.iterrows():
            key = (row['iso'], row['sector_type'])
            output_map[key] = row['total_output']
            
        print(f"  Loaded economic data for {len(output_map)} region-sector-type combinations")
    except Exception as e:
        print(f"  ERROR loading {OUTPUT_FILE}: {e}")
        return

    # --- 4. Accumulate Losses per Region-Sector-Type ---
    print("\n[4/5] Processing loss files (accumulating by Affected Region + Sector Type)...")
    
    # Dictionaries to store aggregated data
    # key = (affected_iso, sector_type)
    region_sector_type_loss_sum: Dict[Tuple[str, str], float] = {}
    region_sector_type_appearance_count: Dict[Tuple[str, str], int] = {}
    
    processed_count = 0
    
    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"
        
        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")
            
            try:
                # LOAD file
                df_loss = pd.read_csv(loss_path)
                
                if 'region' not in df_loss.columns or 'sector' not in df_loss.columns or 'loss' not in df_loss.columns:
                    del df_loss
                    continue

                # 1. EXTRACT AFFECTED REGION ISO from 'region' column
                affected_iso = df_loss['region']
                
                # 2. EXTRACT SECTOR NAME from 'sector' column (e.g., "ARE_Administrative services" -> "Administrative services")
                # Split on first underscore only to preserve sector names with underscores
                df_loss['sector_name'] = df_loss['sector'].str.split('_', n=1).str[1]
                
                # 3. MAP sector_name to sector_type
                df_loss['sector_type'] = df_loss['sector_name'].map(sector_name_to_type)
                
                # Drop rows where mapping failed
                df_loss = df_loss.dropna(subset=['sector_type'])
                
                if len(df_loss) == 0:
                    del df_loss
                    continue

                # 4. CREATE grouping key: (Affected ISO, Sector Type)
                df_loss['key'] = list(zip(affected_iso, df_loss['sector_type']))
                
                # 5. GROUP BY key within this specific simulation file
                # This sums all losses for a specific region-sector-type combination in this specific shock event
                file_agg = df_loss.groupby('key')['loss'].sum()
                
                for key, loss_val in file_agg.items():
                    # Accumulate Total Loss
                    region_sector_type_loss_sum[key] = region_sector_type_loss_sum.get(key, 0.0) + loss_val
                    
                    # Increment Appearance Counter 
                    # (Count +1 for this file because the region-sector-type exists in this file's loss report)
                    region_sector_type_appearance_count[key] = region_sector_type_appearance_count.get(key, 0) + 1
                
                # CLEANUP
                del df_loss
                del file_agg
                gc.collect()
                processed_count += 1
                
            except Exception as e:
                print(f"\n  ERROR processing {folder}: {e}")
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")

    print(f"\n  Completed processing {processed_count} simulation files")
    
    # --- 5. Calculate Intensity Score & Export ---
    print("\n[5/5] Calculating intensity scores and saving...")
    
    results = []
    
    # Iterate over all region-sector-types that suffered loss
    for key in region_sector_type_loss_sum:
        iso, sector_type = key
        total_loss = region_sector_type_loss_sum[key]
        appearance_count = region_sector_type_appearance_count[key]
        
        # Get Economic Output
        # Look up using the AFFECTED region ISO and the Sector Type
        total_output = output_map.get(key, None)
        
        if total_output is None:
            # print(f"  WARNING: No economic output data found for {key}. Skipping.")
            continue
            
        if total_output <= 0:
            # print(f"  WARNING: Zero or negative output for {key}. Skipping.")
            continue
            
        if appearance_count <= 0:
            continue

        # FORMULA: Total Loss / (Total Output * Appearance Counter)
        denominator = total_output * appearance_count
        intensity_score = total_loss / denominator
        
        results.append({
            'iso': iso,
            'sector_type': sector_type,
            'total_loss': total_loss,
            'total_output': total_output,
            'appearance_count': appearance_count,
            'intensity_score': intensity_score
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        # Sort by intensity score descending (highest vulnerability first)
        df_results = df_results.sort_values('intensity_score', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        df_results.to_csv(OUT_FILE, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"DONE! Results saved to {OUT_FILE}")
        print(f"{'=' * 80}")
        
        # Print Summary
        print("\n📊 TOP 10 MOST VULNERABLE REGION-SECTOR-TYPES (BY INTENSITY):")
        print(df_results[['iso', 'sector_type', 'intensity_score', 'total_loss', 'appearance_count']].head(10).to_string(index=True))
        
    else:
        print("\nERROR: No results generated. Check sector mapping and region matching between loss files and output file.")

if __name__ == "__main__":
    main()