import pandas as pd
from pathlib import Path
from scipy import stats  # Added for correlation calculations

FD_FILE = '/home/user/Documents/University/Master Thesis/Databases/Tiva/fd_import_dependency_oecd_tiva.csv'
IMP_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/import_dependency_per_region.csv'
OUT_FILE = 'rank_differences_desc.csv'


def main():
    # 1. Load Data
    fd = pd.read_csv(FD_FILE)
    imp = pd.read_csv(IMP_FILE)

    # 2. Identify Common Regions
    # Ensure column names match your actual CSV headers. 
    # Assuming 'region' is the common key in both files based on your snippet.
    common_regions = sorted(set(fd['iso']).intersection(set(imp['iso'])))

    if not common_regions:
        print("Error: No overlapping regions found between the two datasets.")
        return

    fd_common = fd[fd['iso'].isin(common_regions)].copy()
    imp_common = imp[imp['iso'].isin(common_regions)].copy()

    # 3. Calculate Ranks
    # ascending=False: Highest dependency gets Rank 1
    # method='dense': No gaps in ranking numbers if there are ties
    fd_common['rank_tiva'] = fd_common['final_demand_import_dependency'].rank(
        ascending=False, method='dense'
    ).astype(int)
    
    imp_common['rank_import_dependency'] = imp_common['final_demand_import_dependency'].rank(
        ascending=False, method='dense'
    ).astype(int)

    # 4. Merge Data
    merged = pd.merge(
        fd_common[['iso', 'Final demand area', 'final_demand_import_dependency', 'rank_tiva']],
        imp_common[['iso', 'region_name', 'final_demand_import_dependency', 'rank_import_dependency']],
        on='iso',
        how='inner',
        suffixes=('_tiva', '_import_dependency')
    )

    # 5. Calculate Differences
    merged['rank_difference'] = merged['rank_tiva'] - merged['rank_import_dependency']
    merged['absolute_rank_difference'] = merged['rank_difference'].abs()

    # 6. Calculate Correlation Metrics
    # Spearman's Rho
    spearman_corr, spearman_p = stats.spearmanr(
        merged['rank_tiva'], 
        merged['rank_import_dependency']
    )
    
    # Kendall's Tau
    kendall_tau, kendall_p = stats.kendalltau(
        merged['rank_tiva'], 
        merged['rank_import_dependency']
    )

    # Mean Absolute Difference
    avg_abs_diff = merged['absolute_rank_difference'].mean()
    n_regions = len(merged)

    # 7. Sort Results
    merged = merged.sort_values(
        ['absolute_rank_difference', 'iso'],
        ascending=[False, True]
    ).reset_index(drop=True)

    # 8. Create Summary Rows
    # Row 1: Statistical Correlations
    summary_stats_row = pd.DataFrame([{
        'iso': 'SUMMARY_STATISTICS',
        'Final demand area': f'N={n_regions}',
        'region_name': '',
        'final_demand_import_dependency_tiva': '',
        'final_demand_import_dependency_import_dependency': '',
        'rank_tiva': '',
        'rank_import_dependency': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}',
    }])

    # Row 2: Average Absolute Difference
    avg_row = pd.DataFrame([{
        'iso': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'Final demand area': '',
        'region_name': '',
        'final_demand_import_dependency_tiva': '',
        'final_demand_import_dependency_import_dependency': '',
        'rank_tiva': '',
        'rank_import_dependency': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff,
    }])

    # Combine Summary and Data
    output = pd.concat([summary_stats_row, avg_row, merged], ignore_index=True)

    # 9. Save Output
    output.to_csv(OUT_FILE, index=False)

    # 10. Print Results to Console
    print(f'--- Analysis Complete ---')
    print(f'Overlapping regions analyzed: {n_regions}')
    print(f'Average Absolute Rank Difference: {avg_abs_diff:.2f}')
    print(f'Spearman\'s Rank Correlation (rho): {spearman_corr:.4f} (p-value: {spearman_p:.4e})')
    print(f'Kendall\'s Rank Correlation (tau): {kendall_tau:.4f} (p-value: {kendall_p:.4e})')
    print(f'CSV written to: {Path(OUT_FILE).resolve()}')


if __name__ == '__main__':
    main()