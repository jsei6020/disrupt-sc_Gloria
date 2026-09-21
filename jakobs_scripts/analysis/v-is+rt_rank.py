import pandas as pd
from pathlib import Path


INPUT_FILE_TIME = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-vs_rt.csv'
INPUT_FILE_INTENSITY = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-is_fd.csv'
OUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-intensity_vs_rt_rank_sum.csv'


def main():
    # 1. Load data
    try:
        df_time_raw = pd.read_csv(INPUT_FILE_TIME)
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_FILE_TIME}")
        return

    try:
        df_int_raw = pd.read_csv(INPUT_FILE_INTENSITY)
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_FILE_INTENSITY}")
        return

    print("Loaded time columns:")
    print(df_time_raw.columns.tolist())
    print("\nLoaded intensity columns:")
    print(df_int_raw.columns.tolist())

    # 2. Check required columns
    # From v-vs_rt.csv: iso_3/metric_time_90
    required_time_cols = [
        'iso_3', 'region_name_3', 'metric_time_90'
    ]
    missing_time = [c for c in required_time_cols if c not in df_time_raw.columns]
    if missing_time:
        print(f"Error: Missing required time columns: {missing_time}")
        return

    # From v-is_fd.csv: iso/intensity_score
    # (assuming region_name column is called 'region_name')
    required_int_cols = [
        'iso', 'region_name', 'intensity_score'
    ]
    missing_int = [c for c in required_int_cols if c not in df_int_raw.columns]
    if missing_int:
        print(f"Error: Missing required intensity columns: {missing_int}")
        return

    # 3. Extract time_90 list (unchanged source)
    df_time = df_time_raw[['iso_3', 'region_name_3', 'metric_time_90']].copy()
    df_time.rename(columns={
        'iso_3': 'iso',
        'region_name_3': 'region_name_time'
    }, inplace=True)
    df_time.dropna(subset=['iso', 'metric_time_90'], inplace=True)

    # 4. Extract intensity list
    df_int = df_int_raw[['iso', 'region_name', 'intensity_score']].copy()
    df_int.rename(columns={
        'region_name': 'region_name_intensity'
    }, inplace=True)
    df_int.dropna(subset=['iso', 'intensity_score'], inplace=True)

    # 5. Merge by ISO
    merged = pd.merge(df_int, df_time, on='iso', how='inner')

    if merged.empty:
        print("Error: No overlapping ISO codes between intensity_score and metric_time_90.")
        return

    # 6. One region name column
    merged['region_name'] = merged['region_name_intensity'].combine_first(
        merged['region_name_time']
    )

    # 7. Rank each metric (higher value = rank 1)
    merged['rank_intensity_score'] = (
        merged['intensity_score']
        .rank(ascending=False, method='dense')
        .astype(int)
    )
    merged['rank_metric_time_90'] = (
        merged['metric_time_90']
        .rank(ascending=False, method='dense')
        .astype(int)
    )

    # 8. Combine ranks (analogous to loss_rank + time_rank/2)
    merged['rank_sum_intensity_time'] = (
        merged['rank_intensity_score'] + merged['rank_metric_time_90'] / 2
    )

    # 9. Rank countries by the combined score (smallest = rank 1)
    merged['rank_combined'] = (
        merged['rank_sum_intensity_time']
        .rank(ascending=True, method='dense')
        .astype(int)
    )

    # 10. Sort by combined rank, then ISO
    output = merged[[
        'iso',
        'region_name',
        'intensity_score',
        'metric_time_90',
        'rank_intensity_score',
        'rank_metric_time_90',
        'rank_sum_intensity_time',
        'rank_combined'
    ]].sort_values(
        by=['rank_combined', 'rank_sum_intensity_time', 'iso'],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    # 11. Save output
    output.to_csv(OUT_FILE, index=False)

    # 12. Print summary
    print('--- Rank Sum Analysis Complete (intensity_score × metric_time_90) ---')
    print(f'Regions with both metrics: {len(output)}')
    print(f'Output written to: {Path(OUT_FILE).resolve()}')
    print('\nTop 10 regions by combined rank (lowest combined score):')
    print(output.head(10).to_string(index=False))


if __name__ == '__main__':
    main()