import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final8weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status2.csv"
OUT_FILE = "vulnerability_direct_indirect_ratio_by_region.csv"


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: INDIRECT/DIRECT RATIO BY AFFECTED REGION")
    print("=" * 80)
    
    # --- 1. Load Runs Mapping (Folder -> Disrupted Region-Sector) ---
    print("\n[1/4] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]
    
    # Create a map for fast lookup: folder_name -> disrupted_region_sector
    folder_to_disruption = dict(zip(runs["folder_name"], runs["region_sector"]))
    print(f"  Found {len(folder_to_disruption)} valid simulations")

    # --- 2. Accumulate Direct and Indirect Losses per Affected Region ---
    print("\n[2/4] Processing loss files...")
    
    # Dictionaries to store aggregated data per affected region (ISO)
    region_direct_loss: Dict[str, float] = {}
    region_indirect_loss: Dict[str, float] = {}
    
    processed_count = 0
    
    for idx, row in runs.iterrows():
        folder = row["folder_name"]
        disrupted_region_sector = folder_to_disruption.get(folder)
        
        if not disrupted_region_sector:
            continue
            
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"
        
        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")
            
            try:
                df_loss = pd.read_csv(loss_path)
                
                if 'region' not in df_loss.columns or 'sector' not in df_loss.columns or 'loss' not in df_loss.columns:
                    del df_loss
                    processed_count += 1
                    continue
                
                # DETERMINE Direct vs Indirect based on disruption source
                # Direct: The 'sector' column in loss file MATCHES the disrupted_region_sector of this simulation
                # Indirect: The 'sector' column does NOT match
                df_loss['is_direct'] = df_loss['sector'] == disrupted_region_sector
                
                # AGGREGATE by affected region ('region' column) and direct/indirect flag
                grouped = df_loss.groupby(['region', 'is_direct'])['loss'].sum()
                
                for (affected_region, is_direct), loss_val in grouped.items():
                    if is_direct:
                        region_direct_loss[affected_region] = region_direct_loss.get(affected_region, 0.0) + loss_val
                    else:
                        region_indirect_loss[affected_region] = region_indirect_loss.get(affected_region, 0.0) + loss_val
                
                # CLEANUP
                del df_loss
                del grouped
                gc.collect()
                processed_count += 1
                
            except Exception as e:
                print(f"\n  ERROR processing {folder}: {e}")
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")

    print(f"\n  Completed processing {processed_count} simulation files")
    
    # --- 3. Calculate Ratio & Export ---
    print("\n[3/4] Calculating indirect/direct ratios...")
    
    results = []
    
    # Get all unique regions from both direct and indirect loss dictionaries
    all_regions = set(region_direct_loss.keys()) | set(region_indirect_loss.keys())
    
    for region_iso in all_regions:
        direct_loss = region_direct_loss.get(region_iso, 0.0)
        indirect_loss = region_indirect_loss.get(region_iso, 0.0)
        
        # Calculate Ratio: indirect / direct
        if direct_loss > 0:
            indirect_direct_ratio = indirect_loss / direct_loss
        else:
            indirect_direct_ratio = np.nan
        
        results.append({
            'iso': region_iso,
            'direct_loss': direct_loss,
            'indirect_loss': indirect_loss,
            'indirect_direct_ratio': indirect_direct_ratio
        })
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        # Sort by ratio descending (NaN at bottom)
        df_results = df_results.sort_values('indirect_direct_ratio', ascending=False, na_position='last').reset_index(drop=True)
        
        # Save
        df_results.to_csv(OUT_FILE, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"DONE! Results saved to {OUT_FILE}")
        print(f"{'=' * 80}")
        
        # Print Summary
        print("\n📊 TOP 10 REGIONS BY INDIRECT/DIRECT LOSS RATIO:")
        print(df_results[['iso', 'indirect_direct_ratio', 'direct_loss', 'indirect_loss']].head(10).to_string(index=True))
        
        # Overall Stats
        total_direct = df_results['direct_loss'].sum()
        total_indirect = df_results['indirect_loss'].sum()
        overall_ratio = total_indirect / total_direct if total_direct > 0 else np.nan
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  Total regions analyzed: {len(df_results)}")
        print(f"  Total Direct Loss (all regions): ${total_direct:,.2f}")
        print(f"  Total Indirect Loss (all regions): ${total_indirect:,.2f}")
        print(f"  Overall Indirect/Direct Ratio: {overall_ratio:.4f}")
        
    else:
        print("\nERROR: No results generated.")

if __name__ == "__main__":
    main()