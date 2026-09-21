import pandas as pd
from pathlib import Path
from scipy import stats  # Added for correlation calculations

IMP_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-fdid.csv'
VI_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-is.csv'
OUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/comp_is_fdid.csv'

def main():
    # 1. Load Data
    imp = pd.read_csv(IMP_FILE)
    vi = pd.read_csv(VI_FILE)

    # 2. Calculate Ranks
    # ascending=False means highest value gets Rank 1. 
    # method='dense' ensures no gaps in ranking numbers if there are ties.
    imp['rank_import_dependency'] = imp['final_demand_import_dependency'].rank(ascending=False, method='dense')
    vi['rank_intensity'] = vi['intensity_score'].rank(ascending=False, method='dense')

    # 3. Merge Data
    merged = pd.merge(
        imp[['iso', 'region_name', 'final_demand_import_dependency', 'rank_import_dependency']],
        vi[['iso', 'region_name', 'intensity_score', 'rank_intensity']],
        on='iso',
        how='inner',
        suffixes=('_import_dependency', '_intensity')
    )

    # 4. Clean Data
    # Drop rows where either rank is NaN (cannot correlate or difference NaNs)
    merged = merged.dropna(subset=['rank_import_dependency', 'rank_intensity'])

    # Convert ranks to int for cleaner output (optional, but good for readability)
    merged['rank_import_dependency'] = merged['rank_import_dependency'].astype(int)
    merged['rank_intensity'] = merged['rank_intensity'].astype(int)

    # 5. Calculate Differences
    merged['rank_difference'] = merged['rank_import_dependency'] - merged['rank_intensity']
    merged['absolute_rank_difference'] = merged['rank_difference'].abs()

    # 6. Calculate Correlation Metrics
    # Spearman's Rho
    spearman_corr, spearman_p = stats.spearmanr(
        merged['rank_import_dependency'], 
        merged['rank_intensity']
    )
    
    # Kendall's Tau (often preferred for smaller datasets or many ties)
    kendall_tau, kendall_p = stats.kendalltau(
        merged['rank_import_dependency'], 
        merged['rank_intensity']
    )

    # Mean Absolute Difference (your original metric)
    avg_abs_diff = merged['absolute_rank_difference'].mean()
    n_regions = len(merged)

    # 7. Sort Results
    # Sort by largest absolute difference first, then by ISO code for stability
    merged = merged.sort_values(
        ['absolute_rank_difference', 'iso'],
        ascending=[False, True]
    ).reset_index(drop=True)

    # 8. Create Summary Row
    # We create a DataFrame with the summary stats to prepend to the CSV
    summary_row = pd.DataFrame([{
        'iso': 'SUMMARY_STATISTICS',
        'region_name_import_dependency': f'N={n_regions}',
        'region_name_intensity': '',
        'final_demand_import_dependency': '',
        'intensity_score': '',
        'rank_import_dependency': '',
        'rank_intensity': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}',
        # You can add more columns if needed, or just use the existing ones to store stats
    }])
    
    # Note: To make the summary row clearer, let's add specific columns for stats 
    # if you want them easily parsable, otherwise the string format above works for human reading.
    # For a cleaner academic table, I'll append the stats as separate rows or a header block.
    # Here, I will stick to your previous style but add the Mean Abs Diff back in clearly.
    
    avg_row = pd.DataFrame([{
        'iso': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'region_name_import_dependency': '',
        'region_name_intensity': '',
        'final_demand_import_dependency': '',
        'intensity_score': '',
        'rank_import_dependency': '',
        'rank_intensity': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff,
    }])

    # Combine Summary, Average, and Data
    # Note: Prepending non-data rows to a CSV makes it harder to read back into pandas later.
    # It is often better to save stats separately or print them. 
    # However, keeping your requested format:
    output = pd.concat([summary_row, avg_row, merged], ignore_index=True)

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