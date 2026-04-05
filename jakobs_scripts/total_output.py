import pandas as pd

# --- CONFIG ---
MRIO_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd.pkl"
SECTOR_TABLE_FILE = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/sector_table_g.csv"
OUT_FILE_REGION = "total_region_output.csv"
OUT_FILE_REGION_SECTOR_TYPE = "region_sector_type_total_output.csv"


def main():
    # --- 1. Load Data ---
    print("Loading MRIO file...")
    io_df = pd.read_pickle(MRIO_FILE)

    print("Loading sector table...")
    sector_table = pd.read_csv(SECTOR_TABLE_FILE)

    # Ensure MRIO index is MultiIndex (region, sector)
    if not isinstance(io_df.index, pd.MultiIndex) or io_df.index.nlevels < 2:
        raise ValueError("MRIO row index must be a MultiIndex with (region, sector).")

    # --- 2. Calculate Total Output per Row (Region-Sector) ---
    # Sum all columns for each row to get total output for that specific region-sector
    print("Calculating total output per region-sector...")
    row_sums = io_df.sum(axis=1)
    
    # Convert to DataFrame for easier manipulation
    # Index levels: 0=region, 1=sector
    df_sums = row_sums.reset_index(name='total_output')
    df_sums.columns = ['iso', 'sector', 'total_output']

    # --- 3. Generate Region Totals ---
    print("Generating region totals...")
    region_totals = (
        df_sums.groupby('iso', as_index=False)['total_output']
        .sum()
        .sort_values('total_output', ascending=False)
        .reset_index(drop=True)
    )
    
    # Ensure column order: iso, total_output
    region_totals = region_totals[['iso', 'total_output']]

    # --- 4. Generate Region-Sector-Type Totals ---
    print("Generating region-sector-type totals...")
    
    # Map sector to type using the sector table
    # Sector table expected columns: 'sector', 'type' (based on original script)
    sector_type_map = sector_table.set_index('sector')['type']
    df_sums['sector_type'] = df_sums['sector'].map(sector_type_map)

    # Group by Region (iso) and Sector Type
    rst_totals = (
        df_sums.groupby(['iso', 'sector_type'], as_index=False)['total_output']
        .sum()
        .sort_values('total_output', ascending=False)
        .reset_index(drop=True)
    )

    # Ensure column order: iso, sector_type, total_output
    rst_totals = rst_totals[['iso', 'sector_type', 'total_output']]

    # --- 5. Export ---
    print(f"Saving {OUT_FILE_REGION}...")
    region_totals.to_csv(OUT_FILE_REGION, index=False)

    print(f"Saving {OUT_FILE_REGION_SECTOR_TYPE}...")
    rst_totals.to_csv(OUT_FILE_REGION_SECTOR_TYPE, index=False)

    print("Done!")


if __name__ == "__main__":
    main()