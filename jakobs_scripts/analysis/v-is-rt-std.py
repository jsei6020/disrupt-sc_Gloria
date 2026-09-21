import pandas as pd
from pathlib import Path

INPUT_FILE = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-vs_rt.csv'
OUT_FILE   = '/home/user/Documents/University/Master Thesis/vulnerability_analysis/v-vs_rt_std_loss_time_product.csv'


def main():
    # 1. Load data
    try:
        df_raw = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_FILE}")
        return

    print("Loaded columns:")
    print(df_raw.columns.tolist())

    # 2. Check required columns
    required_cols = [
        'iso_0', 'region_name_0', 'metric_loss_std',
        'iso_3', 'region_name_3', 'metric_time_90'
    ]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        return

    # 3. Extract metric_loss_std list
    df_loss = df_raw[['iso_0', 'region_name_0', 'metric_loss_std']].copy()
    df_loss.rename(columns={
        'iso_0': 'iso',
        'region_name_0': 'region_name_loss'
    }, inplace=True)
    df_loss.dropna(subset=['iso', 'metric_loss_std'], inplace=True)

    # 4. Extract metric_time_90 list
    df_time = df_raw[['iso_3', 'region_name_3', 'metric_time_90']].copy()
    df_time.rename(columns={
        'iso_3': 'iso',
        'region_name_3': 'region_name_time'
    }, inplace=True)
    df_time.dropna(subset=['iso', 'metric_time_90'], inplace=True)

    # 5. Standardize each metric so max = 1
    max_loss = df_loss['metric_loss_std'].max()
    max_time = df_time['metric_time_90'].max()

    if max_loss == 0 or pd.isna(max_loss):
        print("Error: metric_loss_std maximum is zero or missing.")
        return
    if max_time == 0 or pd.isna(max_time):
        print("Error: metric_time_90 maximum is zero or missing.")
        return

    df_loss['metric_loss_std_std'] = df_loss['metric_loss_std'] / max_loss
    df_time['metric_time_90_std'] = df_time['metric_time_90'] / max_time

    # 6. Merge by ISO
    merged = pd.merge(df_loss, df_time, on='iso', how='inner')

    if merged.empty:
        print("Error: No overlapping ISO codes between metric_loss_std and metric_time_90.")
        return

    # 7. Keep a single region name
    merged['region_name'] = merged['region_name_loss'].combine_first(merged['region_name_time'])

    # 8. Add standardized values
    merged['std_product_loss_time'] = (
        merged['metric_loss_std_std'] + merged['metric_time_90_std']
    )

    # 9. Rank by new product (highest product = rank 1)
    merged['rank_std_product_loss_time'] = (
        merged['std_product_loss_time']
        .rank(ascending=False, method='dense')
        .astype(int)
    )

    # Optional: keep component standardized ranks too
    merged['rank_metric_loss_std_std'] = (
        merged['metric_loss_std_std']
        .rank(ascending=False, method='dense')
        .astype(int)
    )
    merged['rank_metric_time_90_std'] = (
        merged['metric_time_90_std']
        .rank(ascending=False, method='dense')
        .astype(int)
    )

    # 10. Final output
    output = merged[[
        'iso',
        'region_name',
        'metric_loss_std',
        'metric_loss_std_std',
        'metric_time_90',
        'metric_time_90_std',
        'std_product_loss_time',
        'rank_std_product_loss_time',
        'rank_metric_loss_std_std',
        'rank_metric_time_90_std'
    ]].sort_values(
        by=['rank_std_product_loss_time', 'std_product_loss_time', 'iso'],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    # 11. Save output
    output.to_csv(OUT_FILE, index=False)

    # 12. Print results
    print('--- Standardized Product Ranking Complete ---')
    print(f'Regions with both metrics: {len(output)}')
    print(f'Max metric_loss_std: {max_loss}')
    print(f'Max metric_time_90: {max_time}')
    print(f'Output written to: {Path(OUT_FILE).resolve()}')
    print('\nTop 20 regions by standardized product:')
    print(output.head(20).to_string(index=False))


if __name__ == '__main__':
    main()