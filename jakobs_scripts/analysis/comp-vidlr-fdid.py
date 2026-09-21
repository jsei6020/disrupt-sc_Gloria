import pandas as pd
from scipy import stats
from pathlib import Path

# File paths
RATIO_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-idlr.csv'
IMP_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-fdid.csv'
OUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/comp_vidlr_fdid.csv'

def main():
    # 1. Load Data
    df_ratio = pd.read_csv(RATIO_FILE)
    df_imp = pd.read_csv(IMP_FILE)

    # 2. Prepare Data for Merging
    # Select relevant columns and rename for clarity
    df_ratio_clean = df_ratio[['iso', 'region_name', 'indirect_direct_ratio']].copy()
    df_imp_clean = df_imp[['iso', 'region_name', 'final_demand_import_dependency']].copy()

    # Merge on ISO code (inner join to keep only overlapping regions)
    df_merged = pd.merge(df_ratio_clean, df_imp_clean, on='iso', suffixes=('_ratio', '_import'))
    
    # Drop rows with missing values in either metric
    df_merged = df_merged.dropna(subset=['indirect_direct_ratio', 'final_demand_import_dependency'])
    
    n_regions = len(df_merged)
    print(f"Analyzing {n_regions} overlapping regions...")

    # 3. Calculate Ranks
    # Ascending=False means highest value gets Rank 1
    # method='dense' ensures no gaps in ranking numbers if there are ties
    df_merged['rank_ratio'] = df_merged['indirect_direct_ratio'].rank(ascending=False, method='dense')
    df_merged['rank_import'] = df_merged['final_demand_import_dependency'].rank(ascending=False, method='dense')

    # Convert to int for cleaner output
    df_merged['rank_ratio'] = df_merged['rank_ratio'].astype(int)
    df_merged['rank_import'] = df_merged['rank_import'].astype(int)

    # 4. Calculate Correlation Metrics
    spearman_corr, spearman_p = stats.spearmanr(df_merged['rank_ratio'], df_merged['rank_import'])
    kendall_tau, kendall_p = stats.kendalltau(df_merged['rank_ratio'], df_merged['rank_import'])

    # 5. Calculate Rank Differences
    df_merged['rank_difference'] = df_merged['rank_ratio'] - df_merged['rank_import']
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
        'region_name_ratio': f'N={n_regions}',
        'indirect_direct_ratio': '',
        'final_demand_import_dependency': '',
        'rank_ratio': '',
        'rank_import': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}'
    }])
    
    avg_row = pd.DataFrame([{
        'iso': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'region_name_ratio': '',
        'indirect_direct_ratio': '',
        'final_demand_import_dependency': '',
        'rank_ratio': '',
        'rank_import': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    }])

    # Combine and save
    final_output = pd.concat([summary_row, avg_row, df_sorted], ignore_index=True)
    final_output.to_csv(OUT_FILE, index=False)
    print(f"Detailed results saved to: {Path(OUT_FILE).resolve()}")

if __name__ == '__main__':
    main()