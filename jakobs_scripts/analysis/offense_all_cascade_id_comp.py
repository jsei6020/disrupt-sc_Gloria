import pandas as pd
from scipy import stats
from pathlib import Path

# File paths - Update these if your local paths differ
CASCADE_FILE = '/home/user/Documents/University/Master Thesis/threat_analysis/loss_rankings_cascade_ratio_all.csv'
RATIO_FILE = '/home/user/Documents/University/Master Thesis/threat_analysis/loss_rankings_top3_avg_direct_indirect_ratio.csv'
OUT_FILE = '/home/user/Documents/University/Master Thesis/threat_analysis/rank_comparison_cascade_vs_top3_ratio.csv'

def main():
    # 1. Load Data
    try:
        df_cascade = pd.read_csv(CASCADE_FILE)
        df_ratio = pd.read_csv(RATIO_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        return

    # 2. Prepare Data for Merging
    # Select relevant columns: Region-Sector Identifier and the Ratio metric
    # For Cascade file: 'region_sector' and 'region_sector_cascade_ratio'
    # For Ratio file: 'Global_RegionSector' and 'Global_Ratio'
    
    df_cascade_clean = df_cascade[['region_sector', 'region_sector_cascade_ratio']].copy()
    df_cascade_clean.rename(columns={'region_sector_cascade_ratio': 'cascade_ratio'}, inplace=True)
    
    df_ratio_clean = df_ratio[['Global_RegionSector', 'Global_Ratio']].copy()
    df_ratio_clean.rename(columns={'Global_RegionSector': 'region_sector', 'Global_Ratio': 'top3_ratio'}, inplace=True)

    # Merge on region_sector (inner join to keep only regions present in BOTH datasets)
    # Note: Ensure string formatting matches (e.g., no extra spaces) if merge yields few results
    df_merged = pd.merge(df_cascade_clean, df_ratio_clean, on='region_sector', how='inner')
    
    # Drop rows with missing values in either metric
    df_merged = df_merged.dropna(subset=['cascade_ratio', 'top3_ratio'])
    
    n_regions = len(df_merged)
    
    if n_regions == 0:
        print("Error: No overlapping region_sectors found between the two files.")
        print("Check if column names match exactly or if there are whitespace issues.")
        return

    print(f"Analyzing {n_regions} overlapping region-sectors...")

    # 3. Calculate Ranks
    # Ascending=False means highest ratio gets Rank 1 (Highest Impact/Risk)
    # method='dense' ensures no gaps in ranking numbers if there are ties
    df_merged['rank_cascade'] = df_merged['cascade_ratio'].rank(ascending=False, method='dense')
    df_merged['rank_top3'] = df_merged['top3_ratio'].rank(ascending=False, method='dense')

    # Convert to int for cleaner output
    df_merged['rank_cascade'] = df_merged['rank_cascade'].astype(int)
    df_merged['rank_top3'] = df_merged['rank_top3'].astype(int)

    # 4. Calculate Correlation Metrics
    spearman_corr, spearman_p = stats.spearmanr(df_merged['rank_cascade'], df_merged['rank_top3'])
    kendall_tau, kendall_p = stats.kendalltau(df_merged['rank_cascade'], df_merged['rank_top3'])

    # 5. Calculate Rank Differences
    df_merged['rank_difference'] = df_merged['rank_cascade'] - df_merged['rank_top3']
    df_merged['absolute_rank_difference'] = df_merged['rank_difference'].abs()
    avg_abs_diff = df_merged['absolute_rank_difference'].mean()

    # 6. Print Results
    print(f"--- Analysis Complete ---")
    print(f"Region-Sectors analyzed: {n_regions}")
    print(f"Average Absolute Rank Difference: {avg_abs_diff:.4f}")
    print(f"Spearman's Rank Correlation (rho): {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Kendall's Rank Correlation (tau): {kendall_tau:.4f} (p-value: {kendall_p:.4e})")
    
    # Optional: Print top 5 mismatches
    print("\nTop 5 Largest Rank Mismatches:")
    top_mismatches = df_merged.nlargest(5, 'absolute_rank_difference')[['region_sector', 'rank_cascade', 'rank_top3', 'absolute_rank_difference']]
    print(top_mismatches.to_string(index=False))

    # 7. Save Detailed Results to CSV
    # Sort by absolute difference to highlight largest mismatches
    df_sorted = df_merged.sort_values(by='absolute_rank_difference', ascending=False).reset_index(drop=True)

    # Create summary rows to mimic the style of your previous example
    summary_row = pd.DataFrame([{
        'region_sector': 'SUMMARY_STATISTICS',
        'cascade_ratio': f'N={n_regions}',
        'top3_ratio': '',
        'rank_cascade': '',
        'rank_top3': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}'
    }])
    
    avg_row = pd.DataFrame([{
        'region_sector': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'cascade_ratio': '',
        'top3_ratio': '',
        'rank_cascade': '',
        'rank_top3': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    }])

    # Combine and save
    final_output = pd.concat([summary_row, avg_row, df_sorted], ignore_index=True)
    final_output.to_csv(OUT_FILE, index=False)
    print(f"\nDetailed results saved to: {Path(OUT_FILE).resolve()}")

if __name__ == '__main__':
    main()