import pandas as pd
from scipy import stats
from pathlib import Path

# File paths
# UPDATE THIS PATH to point to your actual vulnerability_intensity_rankings.csv location
INTENSITY_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/vulnerability_intensity_rankings.csv' 
RATIO_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/vulnerability_total_direct_indirect_total_ratio_by_region.csv'
OUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/rank_comparison_ratio_intensity.csv'

def main():
    # 1. Load Data
    try:
        df_ratio = pd.read_csv(RATIO_FILE)
        df_intensity = pd.read_csv(INTENSITY_FILE)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return

    # 2. Prepare Data for Merging
    # Select relevant columns from the ratio file
    df_ratio_clean = df_ratio[['iso', 'region_name', 'indirect_direct_ratio']].copy()
    
    # Select relevant columns from the intensity file
    # Note: The intensity file also has 'region_name'. We will use the one from ratio as primary, 
    # or handle potential name mismatches by relying on ISO.
    df_intensity_clean = df_intensity[['iso', 'intensity_score']].copy()

    # Merge on ISO code (inner join to keep only overlapping regions)
    df_merged = pd.merge(df_ratio_clean, df_intensity_clean, on='iso', how='inner')
    
    # Drop rows with missing values in either metric
    df_merged = df_merged.dropna(subset=['indirect_direct_ratio', 'intensity_score'])
    
    n_regions = len(df_merged)
    
    if n_regions == 0:
        print("Error: No overlapping regions found between the two datasets.")
        return

    print(f"Analyzing {n_regions} overlapping regions...")

    # 3. Calculate Ranks
    # Ascending=False means highest value gets Rank 1
    # method='dense' ensures no gaps in ranking numbers if there are ties
    df_merged['rank_ratio'] = df_merged['indirect_direct_ratio'].rank(ascending=False, method='dense')
    df_merged['rank_intensity'] = df_merged['intensity_score'].rank(ascending=False, method='dense')

    # Convert to int for cleaner output
    df_merged['rank_ratio'] = df_merged['rank_ratio'].astype(int)
    df_merged['rank_intensity'] = df_merged['rank_intensity'].astype(int)

    # 4. Calculate Correlation Metrics
    spearman_corr, spearman_p = stats.spearmanr(df_merged['rank_ratio'], df_merged['rank_intensity'])
    kendall_tau, kendall_p = stats.kendalltau(df_merged['rank_ratio'], df_merged['rank_intensity'])

    # 5. Calculate Rank Differences
    df_merged['rank_difference'] = df_merged['rank_ratio'] - df_merged['rank_intensity']
    df_merged['absolute_rank_difference'] = df_merged['rank_difference'].abs()
    avg_abs_diff = df_merged['absolute_rank_difference'].mean()

    # 6. Print Results
    print(f"--- Analysis Complete ---")
    print(f"Regions analyzed: {n_regions}")
    print(f"Average Absolute Rank Difference: {avg_abs_diff:.4f}")
    print(f"Spearman's Rank Correlation (rho): {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Kendall's Rank Correlation (tau): {kendall_tau:.4f} (p-value: {kendall_p:.4e})")

    # 7. Save Detailed Results to CSV
    # Sort by absolute difference to highlight largest mismatches
    df_sorted = df_merged.sort_values(by='absolute_rank_difference', ascending=False).reset_index(drop=True)

    # Create summary rows
    summary_row = pd.DataFrame([{
        'iso': 'SUMMARY_STATISTICS',
        'region_name': f'N={n_regions}',
        'indirect_direct_ratio': '',
        'intensity_score': '',
        'rank_ratio': '',
        'rank_intensity': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}'
    }])
    
    avg_row = pd.DataFrame([{
        'iso': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'region_name': '',
        'indirect_direct_ratio': '',
        'intensity_score': '',
        'rank_ratio': '',
        'rank_intensity': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    }])

    # Combine and save
    # Ensure column order is consistent
    final_output = pd.concat([summary_row, avg_row, df_sorted], ignore_index=True)
    
    # Reorder columns for clarity in output
    cols = ['iso', 'region_name', 'indirect_direct_ratio', 'intensity_score', 
            'rank_ratio', 'rank_intensity', 'rank_difference', 'absolute_rank_difference']
    
    # Filter cols to ensure they exist in final_output (in case of unexpected missing cols)
    existing_cols = [c for c in cols if c in final_output.columns]
    final_output = final_output[existing_cols]

    final_output.to_csv(OUT_FILE, index=False)
    print(f"Detailed results saved to: {Path(OUT_FILE).resolve()}")

if __name__ == '__main__':
    main()