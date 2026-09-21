import pandas as pd
from scipy import stats
from pathlib import Path

# File paths
INPUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/extended_vulnerability_analysis.csv' 
OUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/comp_id_is.csv'

def main():
    # 1. Load Data
    try:
        df_raw = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    # 2. Prepare Data
    # The user renamed the second ISO column to 'iso2'.
    # We assume the column names are now roughly:
    # Left side: 'iso', 'region_name', 'indirect_loss_to_output_ratio'
    # Right side: 'iso2', 'region_name' (might be duplicated name, let's check), 'direct_loss_to_output_ratio'
    
    # Inspect columns to ensure we pick the right ones
    cols = df_raw.columns.tolist()
    print(f"Loaded columns: {cols}")
    
    # Define expected column names based on user description and previous structure
    # Note: If 'region_name' appears twice, pandas might have renamed the second one to 'region_name.1' 
    # unless explicitly handled during CSV creation. We will try to find the direct loss column first.
    
    col_iso_left = 'iso'
    col_iso_right = 'iso2'
    
    # Identify metric columns
    col_indirect = 'indirect_loss_to_output_ratio'
    col_direct = 'direct_loss_to_output_ratio'
    
    # Identify region name columns
    # Usually the first 'region_name' belongs to the left (iso), 
    # and the second one (possibly 'region_name.1' or just 'region_name' if duplicates allowed) belongs to right.
    # To be safe, we'll grab the region name associated with the 'iso' column primarily, 
    # but since we are merging, we need to ensure we keep the name for the final output.
    
    # Let's select the necessary columns. 
    # We assume the dataframe has: [unnamed, iso, region_name, indirect_ratio, unnamed, iso2, region_name_2?, direct_ratio]
    
    # Filter for the specific columns we need
    required_cols = [col_iso_left, col_iso_right, col_indirect, col_direct]
    
    # Check if all required columns exist
    missing_cols = [c for c in required_cols if c not in cols]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Please ensure the CSV has columns: 'iso', 'iso2', 'indirect_loss_to_output_ratio', 'direct_loss_to_output_ratio'")
        return

    # Extract relevant columns
    # We also want to keep the region name. Let's assume the first 'region_name' column corresponds to 'iso'.
    # If there is a second region name column for 'iso2', we might need it if the names differ, 
    # but usually ISO codes are unique identifiers. We'll use the left side's region name for simplicity,
    # or merge them if needed.
    
    region_col_left = None
    for c in cols:
        if 'region_name' in c and c != 'region_name.1': # Heuristic to pick the first one
            region_col_left = c
            break
    
    if region_col_left:
        df_subset = df_raw[[col_iso_left, region_col_left, col_iso_right, col_indirect, col_direct]].copy()
        df_subset.rename(columns={region_col_left: 'region_name'}, inplace=True)
    else:
        # Fallback if no region name found
        df_subset = df_raw[[col_iso_left, col_iso_right, col_indirect, col_direct]].copy()
        df_subset['region_name'] = df_subset[col_iso_left] # Use ISO as placeholder

    # 3. Merge Data
    # The data comes from two separate lists. We must merge them on the ISO codes.
    # However, the current structure is wide (one row has both indirect and direct data for DIFFERENT countries potentially).
    # Wait, looking at the raw data provided previously:
    # Row 1: KGZ (Indirect) | SDS (Direct)
    # Row 2: SLV (Indirect) | MRT (Direct)
    # This means the rows are NOT aligned by country. They are two independent lists pasted side-by-side.
    
    # Therefore, we cannot just use the rows as they are. We must split the dataframe into two long-format tables and then merge.
    
    # Create Table 1: Indirect Loss
    df_indirect = df_raw[[col_iso_left, col_indirect]].copy()
    df_indirect.rename(columns={col_iso_left: 'iso', col_indirect: 'indirect_loss_ratio'}, inplace=True)
    df_indirect.dropna(subset=['iso', 'indirect_loss_ratio'], inplace=True)
    
    # Create Table 2: Direct Loss
    df_direct = df_raw[[col_iso_right, col_direct]].copy()
    df_direct.rename(columns={col_iso_right: 'iso', col_direct: 'direct_loss_ratio'}, inplace=True)
    df_direct.dropna(subset=['iso', 'direct_loss_ratio'], inplace=True)
    
    # Merge the two tables on 'iso' (Inner Join)
    # This ensures we only analyze countries that have BOTH indirect and direct data
    df_merged = pd.merge(df_indirect, df_direct, on='iso', how='inner')
    
    # Optional: Add Region Name back if available from the original left table
    # We need to map iso to region_name from the original left side
    if region_col_left:
        df_names = df_raw[[col_iso_left, region_col_left]].drop_duplicates()
        df_names.rename(columns={col_iso_left: 'iso', region_col_left: 'region_name'}, inplace=True)
        df_merged = pd.merge(df_merged, df_names, on='iso', how='left')
    else:
        df_merged['region_name'] = df_merged['iso']

    n_regions = len(df_merged)
    print(f"Analyzing {n_regions} overlapping regions...")
    
    if n_regions == 0:
        print("Error: No overlapping regions found between the direct and indirect lists.")
        return

    # 4. Calculate Ranks
    # Ascending=False means highest loss ratio gets Rank 1
    df_merged['rank_indirect'] = df_merged['indirect_loss_ratio'].rank(ascending=False, method='dense')
    df_merged['rank_direct'] = df_merged['direct_loss_ratio'].rank(ascending=False, method='dense')

    # Convert to int for cleaner output
    df_merged['rank_indirect'] = df_merged['rank_indirect'].astype(int)
    df_merged['rank_direct'] = df_merged['rank_direct'].astype(int)

    # 5. Calculate Correlation Metrics
    spearman_corr, spearman_p = stats.spearmanr(df_merged['rank_indirect'], df_merged['rank_direct'])
    kendall_tau, kendall_p = stats.kendalltau(df_merged['rank_indirect'], df_merged['rank_direct'])

    # 6. Calculate Rank Differences
    df_merged['rank_difference'] = df_merged['rank_indirect'] - df_merged['rank_direct']
    df_merged['absolute_rank_difference'] = df_merged['rank_difference'].abs()
    avg_abs_diff = df_merged['absolute_rank_difference'].mean()

    # 7. Print Results
    print(f"--- Analysis Complete ---")
    print(f"Regions analyzed: {n_regions}")
    print(f"Average Absolute Rank Difference: {avg_abs_diff:.4f}")
    print(f"Spearman's Rank Correlation (rho): {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Kendall's Rank Correlation (tau): {kendall_tau:.4f} (p-value: {kendall_p:.4e})")

    # 8. Save Detailed Results to CSV
    # Sort by absolute difference to highlight largest mismatches
    df_sorted = df_merged.sort_values(by='absolute_rank_difference', ascending=False).reset_index(drop=True)

    # Create summary rows
    summary_row = pd.DataFrame([{
        'iso': 'SUMMARY_STATISTICS',
        'region_name': f'N={n_regions}',
        'indirect_loss_ratio': '',
        'direct_loss_ratio': '',
        'rank_indirect': '',
        'rank_direct': '',
        'rank_difference': f'Spearman Rho: {spearman_corr:.4f}',
        'absolute_rank_difference': f'Kendall Tau: {kendall_tau:.4f}'
    }])
    
    avg_row = pd.DataFrame([{
        'iso': 'AVERAGE_ABSOLUTE_DIFFERENCE',
        'region_name': '',
        'indirect_loss_ratio': '',
        'direct_loss_ratio': '',
        'rank_indirect': '',
        'rank_direct': '',
        'rank_difference': '',
        'absolute_rank_difference': avg_abs_diff
    }])

    # Combine and save
    final_output = pd.concat([summary_row, avg_row, df_sorted], ignore_index=True)
    final_output.to_csv(OUT_FILE, index=False)
    print(f"Detailed results saved to: {Path(OUT_FILE).resolve()}")

if __name__ == '__main__':
    main()