import pandas as pd
from pathlib import Path
from scipy import stats

VI_FILE   = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-is.csv'
VVS_FILE  = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-vs_rt_std_loss_time_product.csv'
OUT_FILE  = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/comp_is_metric_mult_std.csv'


def main():
    # 1. Load Data
    vi = pd.read_csv(VI_FILE)
    vvs = pd.read_csv(VVS_FILE)

    print("VI columns:", vi.columns.tolist())
    print("VVS columns:", vvs.columns.tolist())

    # 2. Check required columns
    # v-is: iso, region_name, intensity_score
    # v-vs_rt: iso1 (or iso), metric_mult_std
    required_vi  = ['iso', 'region_name', 'intensity_score']
    required_vvs = ['iso', 'std_product_loss_time']

    missing_vi  = [c for c in required_vi if c not in vi.columns]
    missing_vvs = [c for c in required_vvs if c not in vvs.columns]

    if missing_vi:
        print(f"Error: v-is.csv is missing columns: {missing_vi}")
        return
    if missing_vvs:
        print(f"Error: v-vs_rt.csv is missing columns: {missing_vvs}")
        return

    # 3. Prepare data
    # Rename iso1 -> iso for merging
    vvs_renamed = vvs.rename(columns={'iso_1': 'iso'})

    # Keep only relevant columns
    vi_sub = vi[['iso', 'region_name', 'intensity_score']].copy()
    vvs_sub = vvs_renamed[['iso', 'std_product_loss_time']].copy()

    # Drop rows with missing essential values
    vi_sub = vi_sub.dropna(subset=['iso', 'intensity_score'])
    vvs_sub = vvs_sub.dropna(subset=['iso', 'std_product_loss_time'])

    # 4. Rank variables (higher value = rank 1)
    vi_sub['rank_intensity'] = vi_sub['intensity_score'].rank(
        ascending=False, method='dense'
    )

    vvs_sub['rank_metric'] = vvs_sub['std_product_loss_time'].rank(
        ascending=False, method='dense'
    )

    # Convert ranks to int for readability
    vi_sub['rank_intensity'] = vi_sub['rank_intensity'].astype(int)
    vvs_sub['rank_metric']   = vvs_sub['rank_metric'].astype(int)

    # 5. Merge on ISO
    merged = pd.merge(
        vi_sub,
        vvs_sub,
        on='iso',
        how='inner'
    )

    # Drop any NaNs in the rank columns, just in case
    merged = merged.dropna(subset=['rank_intensity', 'rank_metric'])

    n_regions = len(merged)
    if n_regions == 0:
        print("Error: No overlapping regions between intensity and metric_mult_std.")
        return

    # 6. Rank differences
    merged['rank_difference'] = merged['rank_intensity'] - merged['rank_metric']
    merged['absolute_rank_difference'] = merged['rank_difference'].abs()
    avg_abs_diff = merged['absolute_rank_difference'].mean()

    # 7. Correlation metrics (Spearman & Kendall on ranks)
    spearman_corr, spearman_p = stats.spearmanr(
        merged['rank_intensity'], merged['rank_metric']
    )
    kendall_tau, kendall_p = stats.kendalltau(
        merged['rank_intensity'], merged['rank_metric']
    )

    # 8. Sort results for output
    merged = merged.sort_values(
        ['absolute_rank_difference', 'iso'],
        ascending=[False, True]
    ).reset_index(drop=True)

    # 9. Summary rows in your usual style
    summary_row = pd.DataFrame([{
        'iso': 'SUMMARY_STATISTICS',
        'region_name': f'N={n_regions}',
        'intensity_score': '',
        'metric_mult_std': '',
        'rank_intensity': '',
        'rank_metric': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}'
    }])

    avg_row = pd.DataFrame([{
        'iso': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'region_name': '',
        'intensity_score': '',
        'metric_mult_std': '',
        'rank_intensity': '',
        'rank_metric': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    }])

    # 10. Combine and save
    output = pd.concat([summary_row, avg_row, merged], ignore_index=True)
    output.to_csv(OUT_FILE, index=False)

    # 11. Console output
    print('--- Analysis Complete (Intensity vs metric_mult_std) ---')
    print(f'Overlapping regions analyzed: {n_regions}')
    print(f'Average Absolute Rank Difference: {avg_abs_diff:.4f}')
    print(f"Spearman's Rank Correlation (rho): {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Kendall's Rank Correlation (tau): {kendall_tau:.4f} (p-value: {kendall_p:.4e})")
    print(f'CSV written to: {Path(OUT_FILE).resolve()}')


if __name__ == '__main__':
    main()