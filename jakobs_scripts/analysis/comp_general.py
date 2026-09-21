import pandas as pd
from scipy import stats
from pathlib import Path



# ============================================================================
# CONFIGURATION - Edit these values for each analysis
# ============================================================================


# File paths
FILE_1 = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-fdid.csv'
FILE_2 = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-is_fd.csv'
OUTPUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/comp_v-fdid_is_fd.csv'


# Column mappings
# For FILE_1: specify the ID column and the metric column to rank
FILE_1_ID_COL = 'iso'          # Column containing region/entity identifier
FILE_1_METRIC_COL = 'final_demand_import_dependency'  # Column to rank


# For FILE_2: specify the ID column and the metric column to rank
FILE_2_ID_COL = 'iso'            # Column containing region/entity identifier
FILE_2_METRIC_COL = 'intensity_score'    # Column to rank


# Ranking direction (True = higher values get better rank, False = lower values get better rank)
RANK_DESCENDING = True


# ============================================================================
# MAIN SCRIPT - No need to edit below this line
# ============================================================================


def main():
    # 1. Load data
    try:
        df1 = pd.read_csv(FILE_1)
        df2 = pd.read_csv(FILE_2)
        print(f"Loaded {len(df1)} rows from {Path(FILE_1).name}")
        print(f"Loaded {len(df2)} rows from {Path(FILE_2).name}")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        return


    # 2. Validate columns exist
    required_cols_1 = [FILE_1_ID_COL, FILE_1_METRIC_COL]
    required_cols_2 = [FILE_2_ID_COL, FILE_2_METRIC_COL]
    
    missing_1 = [col for col in required_cols_1 if col not in df1.columns]
    missing_2 = [col for col in required_cols_2 if col not in df2.columns]
    
    if missing_1:
        print(f"Error: Columns {missing_1} not found in {Path(FILE_1).name}")
        print(f"Available columns: {list(df1.columns)}")
        return
    if missing_2:
        print(f"Error: Columns {missing_2} not found in {Path(FILE_2).name}")
        print(f"Available columns: {list(df2.columns)}")
        return


    # 3. Select and prepare data - drop NA values BEFORE merging
    df1_clean = df1[[FILE_1_ID_COL, FILE_1_METRIC_COL]].copy()
    df1_clean = df1_clean.dropna(subset=[FILE_1_METRIC_COL])
    
    df2_clean = df2[[FILE_2_ID_COL, FILE_2_METRIC_COL]].copy()
    df2_clean = df2_clean.dropna(subset=[FILE_2_METRIC_COL])


    # Rename columns to avoid conflicts if they have the same name
    df1_clean = df1_clean.rename(columns={
        FILE_1_ID_COL: 'id',
        FILE_1_METRIC_COL: 'metric_1'
    })
    
    df2_clean = df2_clean.rename(columns={
        FILE_2_ID_COL: 'id',
        FILE_2_METRIC_COL: 'metric_2'
    })


    # 4. Merge datasets
    df_merged = pd.merge(
        df1_clean, 
        df2_clean, 
        on='id',
        how='inner'
    )
    
    n_regions = len(df_merged)
    if n_regions == 0:
        print("Error: No overlapping regions found after merging.")
        return


    print(f"\n{'='*60}")
    print(f"Analyzing {n_regions} overlapping regions")
    print(f"{'='*60}")


    # 5. Calculate ranks
    df_merged['rank_metric_1'] = df_merged['metric_1'].rank(
        ascending=not RANK_DESCENDING, 
        method='dense'
    ).astype(int)
    
    df_merged['rank_metric_2'] = df_merged['metric_2'].rank(
        ascending=not RANK_DESCENDING, 
        method='dense'
    ).astype(int)


    # 6. Calculate rank correlation metrics
    spearman_corr, spearman_p = stats.spearmanr(
        df_merged['rank_metric_1'], 
        df_merged['rank_metric_2']
    )
    kendall_tau, kendall_p = stats.kendalltau(
        df_merged['rank_metric_1'], 
        df_merged['rank_metric_2']
    )

    # 6b. Calculate regular (Pearson) correlation on the raw metrics
    pearson_corr, pearson_p = stats.pearsonr(
        df_merged['metric_1'],
        df_merged['metric_2']
    )

    # Build LaTeX/Overleaf-ready string for thesis (Pearson only)
    if pearson_p < 0.001:
        pearson_p_str = "p < 0.001"
    else:
        pearson_p_str = f"p = {pearson_p:.3f}"

    # Exactly in the Overleaf style you described
    pearson_thesis_str = (
        f"(Pearson's $r$ = {pearson_corr:.3f}, {pearson_p_str})"
    )


    # 7. Calculate rank differences
    df_merged['rank_difference'] = df_merged['rank_metric_1'] - df_merged['rank_metric_2']
    df_merged['absolute_rank_difference'] = df_merged['rank_difference'].abs()
    avg_abs_diff = df_merged['absolute_rank_difference'].mean()
    max_abs_diff = df_merged['absolute_rank_difference'].max()


    # 8. Print results
    print("\nRank Correlation Analysis:")
    print(f"  Spearman's \u03c1 (ranks):  {spearman_corr:>8.4f}  (p = {spearman_p:.4e})")
    print(f"  Kendall's \u03c4 (ranks):   {kendall_tau:>8.4f}  (p = {kendall_p:.4e})")

    print("\nRegular Correlation Analysis (raw metrics):")
    print(f"  Pearson's r:           {pearson_corr:>8.4f}  (p = {pearson_p:.4e})")

    # Thesis-ready line for direct copy into Overleaf
    print("\nThesis-ready Pearson correlation (copy this into LaTeX):")
    print(pearson_thesis_str)
    
    print("\nRank Difference Statistics:")
    print(f"  Mean Absolute Difference:    {avg_abs_diff:.4f}")
    print(f"  Maximum Absolute Difference: {max_abs_diff:.0f}")
    
    # Interpretation (still based on Spearman p-value)
    if spearman_p < 0.001:
        sig_label = "***"
    elif spearman_p < 0.01:
        sig_label = "**"
    elif spearman_p < 0.05:
        sig_label = "*"
    else:
        sig_label = "n.s."
    print(f"  Significance (Spearman): {sig_label}")


    # 9. Sort by absolute difference for output
    df_sorted = df_merged.sort_values(
        by='absolute_rank_difference', 
        ascending=False
    ).reset_index(drop=True)


    # 10. Create summary rows
    summary_data = []

    summary_data.append({
        'id': '',
        'metric_1':     FILE_1_METRIC_COL,
        'metric_2':     FILE_2_METRIC_COL,
        'rank_metric_1': '',
        'rank_metric_2': '',
        'rank_difference': '',
        'absolute_rank_difference': ''
    })
    summary_data.append({
        'id': 'N_REGIONS',
        'metric_1': n_regions,
        'metric_2': '',
        'rank_metric_1': '',
        'rank_metric_2': '',
        'rank_difference': '',
        'absolute_rank_difference': ''
    })
    
    summary_data.append({
        'id': 'SPEARMAN_RHO_RANKS',
        'metric_1': spearman_corr,
        'metric_2': '',
        'rank_metric_1': '',
        'rank_metric_2': '',
        'rank_difference': f'p={spearman_p:.4e}',
        'absolute_rank_difference': ''
    })
    
    summary_data.append({
        'id': 'KENDALL_TAU_RANKS',
        'metric_1': kendall_tau,
        'metric_2': '',
        'rank_metric_1': '',
        'rank_metric_2': '',
        'rank_difference': f'p={kendall_p:.4e}',
        'absolute_rank_difference': ''
    })

    # New summary row for Pearson on raw metrics
    summary_data.append({
        'id': 'PEARSON_R_RAW',
        'metric_1': pearson_corr,
        'metric_2': '',
        'rank_metric_1': '',
        'rank_metric_2': '',
        'rank_difference': f'p={pearson_p:.4e}',
        'absolute_rank_difference': ''
    })
    
    summary_data.append({
        'id': 'MEAN_ABS_RANK_DIFF',
        'metric_1': '',
        'metric_2': '',
        'rank_metric_1': '',
        'rank_metric_2': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    })
    
    summary_data.append({
        'id': '---',
        'metric_1': '---',
        'metric_2': '---',
        'rank_metric_1': '---',
        'rank_metric_2': '---',
        'rank_difference': '---',
        'absolute_rank_difference': '---'
    })


    summary_df = pd.DataFrame(summary_data)
    
    # 11. Combine and save
    final_output = pd.concat([summary_df, df_sorted], ignore_index=True)
    final_output.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {Path(OUTPUT_FILE).resolve()}")
    print(f"{'='*60}\n")



if __name__ == '__main__':
    main()