import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final8weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status2.csv"
OUTPUT_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/total_region_output.csv"
OUT_FILE = "vulnerability_intensity_rankings.csv"

def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: REGIONAL LOSS INTENSITY")
    print("=" * 80)
    
    # --- 1. Load Runs Mapping ---
    print("\n[1/4] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]
    print(f"  Found {len(runs)} valid simulations")

    # --- 2. Load Economic Output Data ---
    print("\n[2/4] Loading total region output data...")
    # Assuming headers are 'iso' and 'total_output'. 
    # If your CSV has no headers, use: names=['iso', 'total_output'], header=None
    try:
        df_output = pd.read_csv(OUTPUT_FILE)
        # Ensure standard column names for consistency
        if df_output.columns[0] != 'iso':
            df_output.rename(columns={df_output.columns[0]: 'iso'}, inplace=True)
        if df_output.columns[1] != 'total_output':
            df_output.rename(columns={df_output.columns[1]: 'total_output'}, inplace=True)
            
        output_map = df_output.set_index('iso')['total_output'].to_dict()
        print(f"  Loaded economic data for {len(output_map)} regions")
    except Exception as e:
        print(f"  ERROR loading {OUTPUT_FILE}: {e}")
        return

    # --- 3. Accumulate Losses per Region ---
    print("\n[3/4] Processing loss files (accumulating by affected region)...")
    
    # Dictionaries to store aggregated data
    # region_loss_sum[region_iso] = float
    # region_appearance_count[region_iso] = int
    region_loss_sum: Dict[str, float] = {}
    region_appearance_count: Dict[str, int] = {}
    
    processed_count = 0
    
    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"
        
        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")
            
            try:
                # LOAD file
                df_loss = pd.read_csv(loss_path)
                
                # Check if 'region' and 'loss' columns exist
                if 'region' not in df_loss.columns or 'loss' not in df_loss.columns:
                    print(f"\n  WARNING: {folder} missing required columns. Skipping.")
                    del df_loss
                    continue

                # GROUP BY region within this specific simulation file
                # This gives us the total loss suffered by each region in THIS specific shock event
                file_agg = df_loss.groupby('region')['loss'].sum()
                
                for region_iso, loss_val in file_agg.items():
                    # Accumulate Total Loss
                    region_loss_sum[region_iso] = region_loss_sum.get(region_iso, 0.0) + loss_val
                    
                    # Increment Appearance Counter 
                    # (Count +1 for this file because the region exists in this file's loss report)
                    region_appearance_count[region_iso] = region_appearance_count.get(region_iso, 0) + 1
                
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
    
    # --- 4. Calculate Intensity Score & Export ---
    print("\n[4/4] Calculating intensity scores and saving...")
    
    results = []
    
    # Iterate over all regions that suffered loss
    for region_iso in region_loss_sum:
        total_loss = region_loss_sum[region_iso]
        appearance_count = region_appearance_count[region_iso]
        
        # Get Economic Output
        total_output = output_map.get(region_iso, None)
        
        if total_output is None:
            print(f"  WARNING: No economic output data found for region '{region_iso}'. Skipping.")
            continue
            
        if total_output <= 0:
            print(f"  WARNING: Zero or negative output for region '{region_iso}'. Skipping.")
            continue
            
        if appearance_count <= 0:
            continue

        # FORMULA: Total Loss / (Total Output * Appearance Counter)
        # This normalizes loss by economic size and frequency of exposure
        denominator = total_output * appearance_count
        intensity_score = total_loss / denominator
        
        results.append({
            'iso': region_iso,
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
        print("\n📊 TOP 10 MOST VULNERABLE REGIONS (BY INTENSITY):")
        print(df_results[['iso', 'intensity_score', 'total_loss', 'appearance_count']].head(10).to_string(index=True))
        
    else:
        print("\nERROR: No results generated. Check region matching between loss files and output file.")

if __name__ == "__main__":
    main()