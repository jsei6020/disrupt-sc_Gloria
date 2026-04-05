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
OUTPUT_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/region_sector_type_total_output.csv"
OUT_FILE = "vulnerability_direct_indirect_ratio_disrupted_sector_type.csv"


def main():
    print("=" * 80)
    print("VULNERABILITY ANALYSIS: DIRECT vs INDIRECT RATIO (DISRUPTED SECTOR-TYPE)")
    print("=" * 80)
    
    # --- 1. Load Runs Mapping (Folder -> Disrupted Region-Sector) ---
    print("\n[1/6] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]
    
    # Create a map for fast lookup: folder_name -> disrupted_region_sector
    folder_to_disruption = dict(zip(runs["folder_name"], runs["region_sector"]))
    print(f"  Found {len(folder_to_disruption)} valid simulations")

    # --- 2. Load Sector Mapping (Name -> Type) ---
    print("\n[2/6] Loading sector table...")
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

    # --- 3. Load Economic Output Data (for reference/filtering) ---
    print("\n[3/6] Loading region-sector-type output data...")
    try:
        df_output = pd.read_csv(OUTPUT_FILE)
        if df_output.columns[0] != 'iso':
            df_output.rename(columns={df_output.columns[0]: 'iso'}, inplace=True)
        if df_output.columns[1] != 'sector_type':
            df_output.rename(columns={df_output.columns[1]: 'sector_type'}, inplace=True)
        if df_output.columns[2] != 'total_output':
            df_output.rename(columns={df_output.columns[2]: 'total_output'}, inplace=True)
            
        output_map = {}
        for _, row in df_output.iterrows():
            key = (row['iso'], row['sector_type'])
            output_map[key] = row['total_output']
            
        print(f"  Loaded economic data for {len(output_map)} combinations")
    except Exception as e:
        print(f"  ERROR loading {OUTPUT_FILE}: {e}")
        return

    # --- 4. Accumulate Direct and Indirect Losses per Affected Region-Sector-Type ---
    print("\n[4/6] Processing loss files...")
    
    # Dictionaries to store aggregated data
    # Key = (affected_iso, sector_type)
    region_sector_type_direct_loss: Dict[Tuple[str, str], float] = {}
    region_sector_type_indirect_loss: Dict[Tuple[str, str], float] = {}
    region_sector_type_exposure_count: Dict[Tuple[str, str], int] = {}
    
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
                
                # 1. EXTRACT AFFECTED REGION ISO from 'region' column
                affected_iso = df_loss['region']
                
                # 2. EXTRACT SECTOR NAME from 'sector' column (e.g., "USA_Manufacturing" -> "Manufacturing")
                # Split on first underscore only
                df_loss['sector_name'] = df_loss['sector'].str.split('_', n=1).str[1]
                
                # 3. MAP sector_name to sector_type
                df_loss['sector_type'] = df_loss['sector_name'].map(sector_name_to_type)
                
                # Drop rows where mapping failed
                df_loss = df_loss.dropna(subset=['sector_type'])
                
                if len(df_loss) == 0:
                    del df_loss
                    processed_count += 1
                    continue

                # 4. CREATE grouping key: (Affected ISO, Sector Type)
                df_loss['key'] = list(zip(affected_iso, df_loss['sector_type']))
                
                # 5. DETERMINE Direct vs Indirect based on disruption source
                # Direct: The 'sector' column in loss file MATCHES the disrupted_region_sector of this simulation
                # Indirect: The 'sector' column does NOT match
                df_loss['is_direct'] = df_loss['sector'] == disrupted_region_sector
                
                # 6. AGGREGATE
                # Group by key and is_direct flag
                grouped = df_loss.groupby(['key', 'is_direct'])['loss'].sum()
                
                for (key, is_direct), loss_val in grouped.items():
                    # Increment Exposure Count (once per file per key)
                    # We handle count separately to ensure it's +1 per file regardless of direct/indirect split
                    pass 
                
                # Update Counts (once per key per file)
                file_keys = df_loss['key'].unique()
                for key in file_keys:
                    region_sector_type_exposure_count[key] = region_sector_type_exposure_count.get(key, 0) + 1
                
                # Update Loss Sums
                direct_agg = df_loss[df_loss['is_direct']].groupby('key')['loss'].sum()
                indirect_agg = df_loss[~df_loss['is_direct']].groupby('key')['loss'].sum()
                
                for key, loss_val in direct_agg.items():
                    region_sector_type_direct_loss[key] = region_sector_type_direct_loss.get(key, 0.0) + loss_val
                
                for key, loss_val in indirect_agg.items():
                    region_sector_type_indirect_loss[key] = region_sector_type_indirect_loss.get(key, 0.0) + loss_val
                
                # CLEANUP
                del df_loss
                del grouped
                del direct_agg
                del indirect_agg
                gc.collect()
                processed_count += 1
                
            except Exception as e:
                print(f"\n  ERROR processing {folder}: {e}")
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")

    print(f"\n  Completed processing {processed_count} simulation files")
    
    # --- 5. Calculate Ratio & Export ---
    print("\n[5/6] Calculating indirect/direct ratios...")
    
    results = []
    
    # Get all unique keys
    all_keys = set(region_sector_type_direct_loss.keys()) | set(region_sector_type_indirect_loss.keys())
    
    for key in all_keys:
        iso, sector_type = key
        direct_loss = region_sector_type_direct_loss.get(key, 0.0)
        indirect_loss = region_sector_type_indirect_loss.get(key, 0.0)
        exposure_count = region_sector_type_exposure_count.get(key, 0)
        total_output = output_map.get(key, None)
        
        # Calculate Ratio
        if direct_loss > 0:
            indirect_direct_ratio = indirect_loss / direct_loss
        else:
            indirect_direct_ratio = np.nan
        
        results.append({
            'iso': iso,
            'sector_type': sector_type,
            'direct_loss': direct_loss,
            'indirect_loss': indirect_loss,
            'total_output': total_output,
            'exposure_count': exposure_count,
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
        print("\n📊 TOP 10 REGION-SECTOR-TYPES BY INDIRECT/DIRECT RATIO:")
        print(df_results[['iso', 'sector_type', 'indirect_direct_ratio', 'direct_loss', 'indirect_loss']].head(10).to_string(index=True))
        
        # Overall Stats
        total_direct = df_results['direct_loss'].sum()
        total_indirect = df_results['indirect_loss'].sum()
        overall_ratio = total_indirect / total_direct if total_direct > 0 else np.nan
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  Total Region-Sector-Types analyzed: {len(df_results)}")
        print(f"  Total Direct Loss: ${total_direct:,.2f}")
        print(f"  Total Indirect Loss: ${total_indirect:,.2f}")
        print(f"  Overall Indirect/Direct Ratio: {overall_ratio:.4f}")
        
    else:
        print("\nERROR: No results generated.")

if __name__ == "__main__":
    main()