import os
import gc
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict

# --- CONFIG ---
BASE_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global4/final8weeks"
RUNS_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/runs_vs_folders_status2.csv"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
OUT_FILE = "loss_rankings_time_to_impact.csv"

# Threshold: 1 trillion USD = 1,000,000,000 thousands (data is in thousands)
THRESHOLD_1T = 1_000_000_000


def extract_region_sector_from_params(params_path: Path) -> Optional[str]:
    """
    Read parameters.yaml as text and extract the region_sector from 
    disruptions.filter.region_sector using regex.
    """
    try:
        with open(params_path, 'r') as f:
            content = f.read()
        
        pattern = r'filter:\s*\n\s+region_sector:\s*\n\s+-\s+(.+?)(?:\n\s+-|\n\s+\w|\Z)'
        match = re.search(pattern, content, re.MULTILINE)
        
        if match:
            return match.group(1).strip()
        
        pattern2 = r'region_sector:\s*\n?\s*-\s+(.+?)(?:\n|\Z)'
        match2 = re.search(pattern2, content, re.MULTILINE)
        
        if match2:
            return match2.group(1).strip()
        
        return None
    
    except Exception as e:
        print(f"Error reading {params_path}: {e}")
        return None


def compute_time_metrics(df_loss: pd.DataFrame) -> Dict[str, float]:
    """
    Compute time-to-impact metrics from loss_per_region_sector_time.csv
    
    Metrics:
    1. time_to_1t: First timestep when cumulative losses reach 1 trillion (in thousands)
    2. time_to_50pct: First timestep when 50% of total damage is reached
    3. time_to_90pct: First timestep when 90% of total damage is reached (Resilience Score)
    
    Interpretation:
    - Lower time_to_90pct = faster damage accumulation = LESS resilient
    - Higher time_to_90pct = slower damage accumulation = MORE resilient
    """
    # Group by timestep and sum losses across all region-sectors
    loss_by_time = df_loss.groupby('time_step')['loss'].sum().reset_index()
    loss_by_time = loss_by_time.sort_values('time_step')
    
    # Calculate cumulative losses over time
    loss_by_time['cumulative_loss'] = loss_by_time['loss'].cumsum()
    
    # Total loss across all timesteps
    total_loss = loss_by_time['cumulative_loss'].max()
    
    # Metric 1: Time to 1 trillion (1,000,000,000 thousands)
    threshold_mask = loss_by_time['cumulative_loss'] >= THRESHOLD_1T
    if threshold_mask.any():
        time_to_1t = loss_by_time.loc[threshold_mask, 'time_step'].min()
    else:
        time_to_1t = np.nan  # Never reached threshold
    
    # Metric 2: Time to 50% of total damage
    threshold_50 = total_loss * 0.5
    threshold_50_mask = loss_by_time['cumulative_loss'] >= threshold_50
    if threshold_50_mask.any():
        time_to_50pct = loss_by_time.loc[threshold_50_mask, 'time_step'].min()
    else:
        time_to_50pct = np.nan
    
    # Metric 3: Time to 90% of total damage (Resilience Score)
    threshold_90 = total_loss * 0.9
    threshold_90_mask = loss_by_time['cumulative_loss'] >= threshold_90
    if threshold_90_mask.any():
        time_to_90pct = loss_by_time.loc[threshold_90_mask, 'time_step'].min()
    else:
        time_to_90pct = np.nan
    
    return {
        'total_loss': total_loss,
        'time_to_1t': time_to_1t,
        'time_to_50pct': time_to_50pct,
        'time_to_90pct': time_to_90pct,
    }


def main():
    print("=" * 80)
    print("LOSS RANKINGS: TIME TO IMPACT ANALYSIS")
    print("=" * 80)
    
    # --- 1. Load mapping and sector table ---
    print("\n[1/5] Loading runs file...")
    runs = pd.read_csv(RUNS_FILE)
    if "exists_in_global4_final" in runs.columns:
        runs = runs[runs["exists_in_global4_final"] == True]
    print(f"  Found {len(runs)} valid simulations")

    print("\n[2/5] Loading sector table...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)
    sector_type_map = sector_table.set_index("sector")["type"]

    # --- 3. Process each simulation ---
    print("\n[3/5] Processing loss files (one folder at a time)...")
    records = []
    
    for idx, row in runs.iterrows():
        region_sector = row["region_sector"]
        folder = row["folder_name"]
        
        # Split region_sector
        region, sector = region_sector.split("_", 1)
        
        loss_path = Path(BASE_PATH) / folder / "loss_per_region_sector_time.csv"
        
        if loss_path.exists():
            print(f"  [{idx+1:4d}/{len(runs)}] {folder}", end="\r")
            # LOAD entire file
            df_loss = pd.read_csv(loss_path)
            
            # COMPUTE metrics
            metrics = compute_time_metrics(df_loss)
            
            # DELETE from memory immediately
            del df_loss
            gc.collect()
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")
            metrics = {
                'total_loss': np.nan,
                'time_to_1t': np.nan,
                'time_to_50pct': np.nan,
                'time_to_90pct': np.nan,
            }
        
        # Store only the aggregated metrics
        records.append({
            "region": region,
            "sector": sector,
            "region_sector": region_sector,
            "total_loss": metrics['total_loss'],
            "time_to_1t": metrics['time_to_1t'],
            "time_to_50pct": metrics['time_to_50pct'],
            "time_to_90pct": metrics['time_to_90pct'],
        })
    
    print(f"\n  Completed {len(records)} simulations")
    all_rs = pd.DataFrame(records)
    
    # Map sector types
    all_rs["sector_type"] = all_rs["sector"].map(sector_type_map)

    # --- 4. Compute rankings for all 4 categories ---
    print("\n[4/5] Computing rankings...")
    
    # 1) Region-sector ranking (sorted by time_to_90pct - resilience score)
    rs_rank = (
        all_rs[["region_sector", "time_to_90pct", "time_to_50pct", "time_to_1t", "total_loss"]]
        .sort_values("time_to_90pct", ascending=True)
        .reset_index(drop=True)
    )
    rs_rank["region_sector_n_runs"] = 1
    rs_rank.columns = [
        "region_sector",
        "region_sector_time_to_90pct",
        "region_sector_time_to_50pct",
        "region_sector_time_to_1t",
        "region_sector_total_loss",
        "region_sector_n_runs",
    ]

    # 2) Region ranking (average time metrics per region)
    by_region = (
        all_rs.groupby("region", as_index=False)
        .agg(
            total_loss=("total_loss", "sum"),
            avg_time_to_1t=("time_to_1t", "mean"),
            avg_time_to_50pct=("time_to_50pct", "mean"),
            avg_time_to_90pct=("time_to_90pct", "mean"),
            n_runs=("time_to_90pct", "size"),
        )
    )
    by_region = (
        by_region.sort_values("avg_time_to_90pct", ascending=True)
        .reset_index(drop=True)
    )
    by_region = by_region[[
        "region", 
        "avg_time_to_90pct",
        "avg_time_to_50pct",
        "avg_time_to_1t",
        "total_loss",
        "n_runs"
    ]]
    by_region.columns = [
        "region", 
        "region_avg_time_to_90pct",
        "region_avg_time_to_50pct",
        "region_avg_time_to_1t",
        "region_total_loss",
        "region_n_runs"
    ]

    # 3) Sector ranking
    by_sector = (
        all_rs.groupby("sector", as_index=False)
        .agg(
            total_loss=("total_loss", "sum"),
            avg_time_to_1t=("time_to_1t", "mean"),
            avg_time_to_50pct=("time_to_50pct", "mean"),
            avg_time_to_90pct=("time_to_90pct", "mean"),
            n_runs=("time_to_90pct", "size"),
        )
    )
    by_sector = (
        by_sector.sort_values("avg_time_to_90pct", ascending=True)
        .reset_index(drop=True)
    )
    by_sector = by_sector[[
        "sector", 
        "avg_time_to_90pct",
        "avg_time_to_50pct",
        "avg_time_to_1t",
        "total_loss",
        "n_runs"
    ]]
    by_sector.columns = [
        "sector", 
        "sector_avg_time_to_90pct",
        "sector_avg_time_to_50pct",
        "sector_avg_time_to_1t",
        "sector_total_loss",
        "sector_n_runs"
    ]

    # 4) Sector-type ranking
    by_sector_type = (
        all_rs.groupby("sector_type", as_index=False)
        .agg(
            total_loss=("total_loss", "sum"),
            avg_time_to_1t=("time_to_1t", "mean"),
            avg_time_to_50pct=("time_to_50pct", "mean"),
            avg_time_to_90pct=("time_to_90pct", "mean"),
            n_runs=("time_to_90pct", "size"),
        )
    )
    by_sector_type = (
        by_sector_type.sort_values("avg_time_to_90pct", ascending=True)
        .reset_index(drop=True)
    )
    by_sector_type = by_sector_type[[
        "sector_type", 
        "avg_time_to_90pct",
        "avg_time_to_50pct",
        "avg_time_to_1t",
        "total_loss",
        "n_runs"
    ]]
    by_sector_type.columns = [
        "sector_type", 
        "sector_type_avg_time_to_90pct",
        "sector_type_avg_time_to_50pct",
        "sector_type_avg_time_to_1t",
        "sector_type_total_loss",
        "sector_type_n_runs"
    ]

    # --- 5. Combine and export ---
    print("\n[5/5] Writing output file...")
    
    max_len = max(len(rs_rank), len(by_region), len(by_sector), len(by_sector_type))
    
    rs_rank = rs_rank.reindex(range(max_len))
    by_region = by_region.reindex(range(max_len))
    by_sector = by_sector.reindex(range(max_len))
    by_sector_type = by_sector_type.reindex(range(max_len))

    final_df = pd.DataFrame({
        # Region-Sector Rankings
        "region_sector": rs_rank["region_sector"],
        "region_sector_time_to_90pct": rs_rank["region_sector_time_to_90pct"],
        "region_sector_time_to_50pct": rs_rank["region_sector_time_to_50pct"],
        "region_sector_time_to_1t": rs_rank["region_sector_time_to_1t"],
        "region_sector_total_loss": rs_rank["region_sector_total_loss"],
        "region_sector_n_runs": rs_rank["region_sector_n_runs"],
        
        "sep1": pd.NA,
        
        # Region Rankings
        "region": by_region["region"],
        "region_time_to_90pct": by_region["region_avg_time_to_90pct"],
        "region_time_to_50pct": by_region["region_avg_time_to_50pct"],
        "region_time_to_1t": by_region["region_avg_time_to_1t"],
        "region_total_loss": by_region["region_total_loss"],
        "region_n_runs": by_region["region_n_runs"],
        
        "sep2": pd.NA,
        
        # Sector Rankings
        "sector": by_sector["sector"],
        "sector_time_to_90pct": by_sector["sector_avg_time_to_90pct"],
        "sector_time_to_50pct": by_sector["sector_avg_time_to_50pct"],
        "sector_time_to_1t": by_sector["sector_avg_time_to_1t"],
        "sector_total_loss": by_sector["sector_total_loss"],
        "sector_n_runs": by_sector["sector_n_runs"],
        
        "sep3": pd.NA,
        
        # Sector-Type Rankings
        "sector_type": by_sector_type["sector_type"],
        "sector_type_time_to_90pct": by_sector_type["sector_type_avg_time_to_90pct"],
        "sector_type_time_to_50pct": by_sector_type["sector_type_avg_time_to_50pct"],
        "sector_type_time_to_1t": by_sector_type["sector_type_avg_time_to_1t"],
        "sector_type_total_loss": by_sector_type["sector_type_total_loss"],
        "sector_type_n_runs": by_sector_type["sector_type_n_runs"],
    })

    final_df.to_csv(OUT_FILE, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"DONE! Results saved to {OUT_FILE}")
    print(f"{'=' * 80}")
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"  Total simulations processed: {len(all_rs)}")
    print(f"  Average total loss: ${all_rs['total_loss'].mean():,.2f} thousands")
    print(f"  Average time to 1T: {all_rs['time_to_1t'].mean():.2f} timesteps")
    print(f"  Average time to 50%: {all_rs['time_to_50pct'].mean():.2f} timesteps")
    print(f"  Average time to 90%: {all_rs['time_to_90pct'].mean():.2f} timesteps")
    
    print(f"\n  Fastest recovery (Top 5 - fastest to 90% damage):")
    top5 = all_rs.nsmallest(5, 'time_to_90pct')[['region_sector', 'time_to_90pct', 'time_to_50pct', 'time_to_1t', 'total_loss']]
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"    {i}. {row['region_sector']}")
        print(f"       90%: timestep {row['time_to_90pct']:.2f} | 50%: {row['time_to_50pct']:.2f} | 1T: {row['time_to_1t']:.2f}")
    
    print(f"\n  Slowest recovery (Bottom 5 - slowest to 90% damage):")
    bottom5 = all_rs.nlargest(5, 'time_to_90pct')[['region_sector', 'time_to_90pct', 'time_to_50pct', 'time_to_1t', 'total_loss']]
    for i, (_, row) in enumerate(bottom5.iterrows(), 1):
        print(f"    {i}. {row['region_sector']}")
        print(f"       90%: timestep {row['time_to_90pct']:.2f} | 50%: {row['time_to_50pct']:.2f} | 1T: {row['time_to_1t']:.2f}")


if __name__ == "__main__":
    main()