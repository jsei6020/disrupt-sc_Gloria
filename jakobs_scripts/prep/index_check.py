import pandas as pd
import itertools

# Paths from your earlier script
MRIO_PATH = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_adjusted.pkl"
REGIONS_PATH = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/regions_table.csv"
SECTORS_PATH = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/sector_table_g.csv"

print("Loading region and sector tables...")
regions = pd.read_csv(REGIONS_PATH)        # columns: Lfd_Nr, Region_acronyms, Region_names
sectors = pd.read_csv(SECTORS_PATH)        # columns: id, type, sector

region_codes = regions["Region_acronyms"].tolist()
sector_names = sectors["sector"].tolist()

print(f"Regions: {len(region_codes)}, Sectors: {len(sector_names)}")
print(f"Expected core region-sector combinations: {len(region_codes) * len(sector_names)}")

print("\nLoading MRIO...")
mrio = pd.read_pickle(MRIO_PATH)

if not isinstance(mrio.index, pd.MultiIndex) or mrio.index.nlevels < 2:
    raise ValueError("Row index must be a MultiIndex (region, sector).")
if not isinstance(mrio.columns, pd.MultiIndex) or mrio.columns.nlevels < 2:
    raise ValueError("Column index must be a MultiIndex (region, sector_or_fd).")

row_regions = mrio.index.get_level_values(0)
row_sectors = mrio.index.get_level_values(1)
col_regions = mrio.columns.get_level_values(0)
col_sectors = mrio.columns.get_level_values(1)

print(f"MRIO shape: {mrio.shape}")
print(f"Unique row regions: {row_regions.nunique()}, unique row sectors: {row_sectors.nunique()}")
print(f"Unique col regions: {col_regions.nunique()}, unique col sectors: {col_sectors.nunique()}")

# --- Check for missing region codes in rows/cols ---
missing_row_regions = sorted(set(region_codes) - set(row_regions))
missing_col_regions = sorted(set(region_codes) - set(col_regions))

print("\nMissing region codes in rows:", missing_row_regions)
print("Missing region codes in cols:", missing_col_regions)

# --- Check for missing sector names in rows/cols ---
missing_row_sectors = sorted(set(sector_names) - set(row_sectors))
missing_col_sectors = sorted(set(sector_names) - set(col_sectors))

print("\nMissing sector names in rows:", missing_row_sectors)
print("Missing sector names in cols:", missing_col_sectors)

# --- Check missing (region, sector) pairs in rows ---
expected_pairs = set(itertools.product(region_codes, sector_names))
mrio_row_pairs = set(zip(row_regions, row_sectors))
mrio_col_pairs = set(zip(col_regions, col_sectors))

missing_row_pairs = sorted(expected_pairs - mrio_row_pairs)
missing_col_pairs = sorted(expected_pairs - mrio_col_pairs)

print(f"\nNumber of missing (region, sector) pairs in rows: {len(missing_row_pairs)}")
print(f"Number of missing (region, sector) pairs in cols: {len(missing_col_pairs)}")

# If there are any, inspect a few
print("\nFirst 20 missing row pairs:")
for p in missing_row_pairs[:20]:
    print("  ROW missing:", p)

print("\nFirst 20 missing col pairs:")
for p in missing_col_pairs[:20]:
    print("  COL missing:", p)