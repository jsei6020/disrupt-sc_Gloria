import pandas as pd
from scipy import stats
from pathlib import Path

# File paths
# Update these paths to match your actual data files
RATIO_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-vs_rt.csv'
TOTAL_OUTPUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/total_region_output.csv'
OUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/comp_v-rt_to.csv'

def main():
    # 1. Load Data
    try:
        df_ratio = pd.read_csv(RATIO_FILE)
        df_output = pd.read_csv(TOTAL_OUTPUT_FILE)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return

    # 2. Prepare Data for Merging
    # Select relevant columns
    # From first file: iso_3 and metric_time_90
    df_ratio_clean = df_ratio[['iso_3', 'metric_time_90']].copy()
    
    # From second file: iso and total_output
    df_output_clean = df_output[['iso', 'total_output']].copy()

    # Merge on ISO code
    # left_on='iso_3' and right_on='iso' handles the different column names
    df_merged = pd.merge(df_ratio_clean, df_output_clean, left_on='iso_3', right_on='iso', how='inner')
    
    # Drop rows with missing values in either metric
    df_merged = df_merged.dropna(subset=['metric_time_90', 'total_output'])
    
    n_regions = len(df_merged)
    if n_regions == 0:
        print("No overlapping regions found after merging.")
        return

    print(f"Analyzing {n_regions} overlapping regions...")

    # 3. Calculate Ranks
    # Ascending=False means highest value gets Rank 1
    # method='dense' ensures no gaps in ranking numbers if there are ties
    df_merged['rank_metric90'] = df_merged['metric_time_90'].rank(ascending=False, method='dense')
    df_merged['rank_output'] = df_merged['total_output'].rank(ascending=False, method='dense')

    # Convert to int for cleaner output
    df_merged['rank_metric90'] = df_merged['rank_metric90'].astype(int)
    df_merged['rank_output'] = df_merged['rank_output'].astype(int)

    # 4. Calculate Correlation Metrics
    spearman_corr, spearman_p = stats.spearmanr(df_merged['rank_metric90'], df_merged['rank_output'])
    kendall_tau, kendall_p = stats.kendalltau(df_merged['rank_metric90'], df_merged['rank_output'])

    # 5. Calculate Rank Differences
    df_merged['rank_difference'] = df_merged['rank_metric90'] - df_merged['rank_output']
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
        'iso_3': 'SUMMARY_STATISTICS',
        'iso': '', # Empty because this row is meta-data
        'metric_time_90': '',
        'total_output': '',
        'rank_metric90': '',
        'rank_output': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}'
    }])
    
    avg_row = pd.DataFrame([{
        'iso_3': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'iso': '',
        'metric_time_90': '',
        'total_output': '',
        'rank_metric90': '',
        'rank_output': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    }])

    # Combine and save
    final_output = pd.concat([summary_row, avg_row, df_sorted], ignore_index=True)
    final_output.to_csv(OUT_FILE, index=False)
    print(f"Detailed results saved to: {Path(OUT_FILE).resolve()}")

if __name__ == '__main__':
    main()