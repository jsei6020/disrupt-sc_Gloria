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
OUT_FILE = "loss_rankings_direct_indirect_ratio.csv"


def compute_loss_metrics(df_loss: pd.DataFrame, disrupted_region_sector: str) -> Dict[str, float]:
    """
    Compute loss metrics from a fully loaded loss_per_region_sector_time.csv dataframe.
    
    Direct loss: all rows where sector matches the disrupted sector
    Indirect loss: all other rows (cascading effects through the network)
    """
    region, sector = disrupted_region_sector.split("_", 1)
    
    # Total loss: sum of all losses in the file
    total_loss = df_loss['loss'].sum()
    
    # Direct loss: where sector column matches the disrupted sector
    is_direct = df_loss['sector'].str.endswith(f"_{sector}", na=False)
    direct_loss = df_loss.loc[is_direct, 'loss'].sum()
    
    # Indirect loss: everything else (cascading losses through the network)
    indirect_loss = df_loss.loc[~is_direct, 'loss'].sum()
    
    # Ratio of indirect to direct losses
    indirect_direct_ratio = indirect_loss / direct_loss if direct_loss > 0 else np.nan
    
    return {
        'total_loss': total_loss,
        'direct_loss': direct_loss,
        'indirect_loss': indirect_loss,
        'indirect_direct_ratio': indirect_direct_ratio
    }


def main():
    print("=" * 80)
    print("LOSS RANKINGS: DIRECT vs INDIRECT ANALYSIS")
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
            metrics = compute_loss_metrics(df_loss, region_sector)
            
            # DELETE from memory immediately
            del df_loss
            gc.collect()
        else:
            print(f"  [{idx+1:4d}/{len(runs)}] WARNING: {folder} - file not found")
            metrics = {
                'total_loss': np.nan,
                'direct_loss': np.nan,
                'indirect_loss': np.nan,
                'indirect_direct_ratio': np.nan
            }
        
        # Store only the aggregated metrics
        records.append({
            "region": region,
            "sector": sector,
            "region_sector": region_sector,
            "total_loss": metrics['total_loss'],
            "direct_loss": metrics['direct_loss'],
            "indirect_loss": metrics['indirect_loss'],
            "indirect_direct_ratio": metrics['indirect_direct_ratio'],
        })
    
    print(f"\n  Completed {len(records)} simulations")
    all_rs = pd.DataFrame(records)
    
    # Map sector types
    all_rs["sector_type"] = all_rs["sector"].map(sector_type_map)

    # --- 4. Compute rankings for all 4 categories ---
    print("\n[4/5] Computing rankings...")
    
    # 1) Region-sector ranking (sorted by indirect/direct ratio)
    rs_rank = (
        all_rs[["region_sector", "indirect_direct_ratio", "total_loss", "direct_loss", "indirect_loss"]]
        .sort_values("indirect_direct_ratio", ascending=False)
        .reset_index(drop=True)
    )
    rs_rank["region_sector_n_runs"] = 1
    rs_rank.columns = [
        "region_sector",
        "region_sector_indirect_direct_ratio",
        "region_sector_total_loss",
        "region_sector_direct_loss",
        "region_sector_indirect_loss",
        "region_sector_n_runs",
    ]

    # 2) Region ranking
    by_region = (
        all_rs.groupby("region", as_index=False)
        .agg(
            total_loss=("total_loss", "sum"),
            direct_loss=("direct_loss", "sum"),
            indirect_loss=("indirect_loss", "sum"),
            n_runs=("indirect_direct_ratio", "size"),
        )
    )
    by_region["region_indirect_direct_ratio"] = (
        by_region["indirect_loss"] / by_region["direct_loss"]
    )
    by_region = (
        by_region.sort_values("region_indirect_direct_ratio", ascending=False)
        .reset_index(drop=True)
    )
    by_region = by_region[[
        "region", 
        "region_indirect_direct_ratio", 
        "total_loss",
        "direct_loss",
        "indirect_loss",
        "n_runs"
    ]]
    by_region.columns = [
        "region", 
        "region_indirect_direct_ratio", 
        "region_total_loss",
        "region_direct_loss",
        "region_indirect_loss",
        "region_n_runs"
    ]

    # 3) Sector ranking
    by_sector = (
        all_rs.groupby("sector", as_index=False)
        .agg(
            total_loss=("total_loss", "sum"),
            direct_loss=("direct_loss", "sum"),
            indirect_loss=("indirect_loss", "sum"),
            n_runs=("indirect_direct_ratio", "size"),
        )
    )
    by_sector["sector_indirect_direct_ratio"] = (
        by_sector["indirect_loss"] / by_sector["direct_loss"]
    )
    by_sector = (
        by_sector.sort_values("sector_indirect_direct_ratio", ascending=False)
        .reset_index(drop=True)
    )
    by_sector = by_sector[[
        "sector", 
        "sector_indirect_direct_ratio", 
        "total_loss",
        "direct_loss",
        "indirect_loss",
        "n_runs"
    ]]
    by_sector.columns = [
        "sector", 
        "sector_indirect_direct_ratio", 
        "sector_total_loss",
        "sector_direct_loss",
        "sector_indirect_loss",
        "sector_n_runs"
    ]

    # 4) Sector-type ranking
    by_sector_type = (
        all_rs.groupby("sector_type", as_index=False)
        .agg(
            total_loss=("total_loss", "sum"),
            direct_loss=("direct_loss", "sum"),
            indirect_loss=("indirect_loss", "sum"),
            n_runs=("indirect_direct_ratio", "size"),
        )
    )
    by_sector_type["sector_type_indirect_direct_ratio"] = (
        by_sector_type["indirect_loss"] / by_sector_type["direct_loss"]
    )
    by_sector_type = (
        by_sector_type.sort_values("sector_type_indirect_direct_ratio", ascending=False)
        .reset_index(drop=True)
    )
    by_sector_type = by_sector_type[[
        "sector_type", 
        "sector_type_indirect_direct_ratio", 
        "total_loss",
        "direct_loss",
        "indirect_loss",
        "n_runs"
    ]]
    by_sector_type.columns = [
        "sector_type", 
        "sector_type_indirect_direct_ratio", 
        "sector_type_total_loss",
        "sector_type_direct_loss",
        "sector_type_indirect_loss",
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
        "region_sector_indirect_direct_ratio": rs_rank["region_sector_indirect_direct_ratio"],
        "region_sector_total_loss": rs_rank["region_sector_total_loss"],
        "region_sector_direct_loss": rs_rank["region_sector_direct_loss"],
        "region_sector_indirect_loss": rs_rank["region_sector_indirect_loss"],
        "region_sector_n_runs": rs_rank["region_sector_n_runs"],
        
        "sep1": pd.NA,
        
        # Region Rankings
        "region": by_region["region"],
        "region_indirect_direct_ratio": by_region["region_indirect_direct_ratio"],
        "region_total_loss": by_region["region_total_loss"],
        "region_direct_loss": by_region["region_direct_loss"],
        "region_indirect_loss": by_region["region_indirect_loss"],
        "region_n_runs": by_region["region_n_runs"],
        
        "sep2": pd.NA,
        
        # Sector Rankings
        "sector": by_sector["sector"],
        "sector_indirect_direct_ratio": by_sector["sector_indirect_direct_ratio"],
        "sector_total_loss": by_sector["sector_total_loss"],
        "sector_direct_loss": by_sector["sector_direct_loss"],
        "sector_indirect_loss": by_sector["sector_indirect_loss"],
        "sector_n_runs": by_sector["sector_n_runs"],
        
        "sep3": pd.NA,
        
        # Sector-Type Rankings
        "sector_type": by_sector_type["sector_type"],
        "sector_type_indirect_direct_ratio": by_sector_type["sector_type_indirect_direct_ratio"],
        "sector_type_total_loss": by_sector_type["sector_type_total_loss"],
        "sector_type_direct_loss": by_sector_type["sector_type_direct_loss"],
        "sector_type_indirect_loss": by_sector_type["sector_type_indirect_loss"],
        "sector_type_n_runs": by_sector_type["sector_type_n_runs"],
    })

    final_df.to_csv(OUT_FILE, index=False)
    
    print(f"\n{'=' * 80}")
    print(f"DONE! Results saved to {OUT_FILE}")
    print(f"{'=' * 80}")
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"  Total simulations processed: {len(all_rs)}")
    print(f"  Total loss (all sims): ${all_rs['total_loss'].sum():,.2f}")
    print(f"  Total direct loss: ${all_rs['direct_loss'].sum():,.2f}")
    print(f"  Total indirect loss: ${all_rs['indirect_loss'].sum():,.2f}")
    
    overall_ratio = all_rs['indirect_loss'].sum() / all_rs['direct_loss'].sum() if all_rs['direct_loss'].sum() > 0 else np.nan
    print(f"  Overall indirect/direct ratio: {overall_ratio:.4f}")
    
    print(f"\n  🔝 Top 5 region-sectors by indirect/direct ratio:")
    top5 = all_rs.nlargest(5, 'indirect_direct_ratio')[['region_sector', 'indirect_direct_ratio', 'total_loss', 'direct_loss', 'indirect_loss']]
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"    {i}. {row['region_sector']}")
        print(f"       Ratio: {row['indirect_direct_ratio']:.4f} | Direct: ${row['direct_loss']:,.2f} | Indirect: ${row['indirect_loss']:,.2f}")


if __name__ == "__main__":
    main()