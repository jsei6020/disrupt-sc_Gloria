import pandas as pd
import os

# ==========================================
# FILEPATHS
# ==========================================
filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/"
SECTORS_FILE = filepath + "sector_table_g.csv"
REGIONS_FILE = filepath + "regions_table.csv"
MRIO_FILE = filepath + "mrio_va_fd.pkl"

# ==========================================
# CONFIGURATION
# ==========================================
N_REGIONS = 164
N_SECTORS = 120
N_VA_TYPES = 6
N_FD_TYPES = 6

N_INDUSTRIES = N_REGIONS * N_SECTORS  # 19,680
N_VA_ROWS = N_REGIONS * N_VA_TYPES    # 984
N_FD_COLS = N_REGIONS * N_FD_TYPES    # 984

# ==========================================
# LOAD SECTOR NAMES FROM CSV
# ==========================================
# Column 2 contains the sector names (based on your CSV structure: id, type, sector)
sectors_df = pd.read_csv(SECTORS_FILE, header=None, skiprows=1)
new_sectors = sectors_df[2].tolist()  # Column index 2 = "sector" column
print(f"Loaded {len(new_sectors)} sector names from sector_table_g.csv")

# ==========================================
# LOAD REGIONS FROM CSV
# ==========================================
regions_df = pd.read_csv(REGIONS_FILE, header=None, skiprows=1)
regions = regions_df[1].tolist()  # Column index 1 = Region acronyms
print(f"Loaded {len(regions)} regions from regions_table.csv")

# ==========================================
# LOAD EXISTING MRIO FILE
# ==========================================
print("Loading mrio_va_fd.pkl...")
mrio_va_fd = pd.read_pickle(MRIO_FILE)
print(f"Current shape: {mrio_va_fd.shape}")
print(f"Expected shape: ({N_INDUSTRIES + N_VA_ROWS}, {N_INDUSTRIES + N_FD_COLS})")

# ==========================================
# CREATE NEW MULTIINDEX FOR INDUSTRIES
# ==========================================
# Create new MultiIndex for the industry part: each region paired with every new sector name
industry_multi_index = pd.MultiIndex.from_product(
    [regions, new_sectors], 
    names=["region", "sector"]
)
print(f"Industry MultiIndex created: {len(industry_multi_index)} entries")

# ==========================================
# UPDATE INDEX (ROWS)
# ==========================================
print("Updating row index...")

# Get the VA part index (keep unchanged)
va_index = mrio_va_fd.index[N_INDUSTRIES:]

# Combine new industry index with existing VA index
new_index = industry_multi_index.append(va_index)
mrio_va_fd.index = new_index

# ==========================================
# UPDATE COLUMNS
# ==========================================
print("Updating column index...")

# Get the FD part columns (keep unchanged)
fd_columns = mrio_va_fd.columns[N_INDUSTRIES:]

# Combine new industry columns with existing FD columns
new_columns = industry_multi_index.append(fd_columns)
mrio_va_fd.columns = new_columns

# ==========================================
# VERIFY
# ==========================================
print("\n=== Verification ===")
print(f"Final shape: {mrio_va_fd.shape}")
print(f"\nIndex levels: {mrio_va_fd.index.names}")
print(f"Columns levels: {mrio_va_fd.columns.names}")

print(f"\nFirst 5 index entries (Industries):")
print(mrio_va_fd.index[:5])

print(f"\nLast 5 index entries (VA types):")
print(mrio_va_fd.index[-5:])

print(f"\nFirst 5 column entries (Industries):")
print(mrio_va_fd.columns[:5])

print(f"\nLast 5 column entries (FD types):")
print(mrio_va_fd.columns[-5:])

# ==========================================
# EXPORT
# ==========================================
output_file = filepath + "mrio_va_fd2.pkl"
print(f"\nSaving updated table to {output_file}")
mrio_va_fd.to_pickle(output_file)

print(mrio_va_fd)