import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
basepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/"
YEAR = 2023
TAG  = "20251217_120secMother_AllCountries_002"
MRIO_DIR = Path(basepath, "GLORIA_MRIOs_60_2023")

REGIONS_FILE = Path(basepath, "IO-Table/regions_table.csv")
SECTORS_FILE = Path(basepath, "IO-Table/sector_table_g.csv")

R = 164
I = 120
P = 120
FD = 6
ROWS_PER_REGION = I + P

target_region = "XAF"
target_sector = "Air transport"

def t_results_path(m: int) -> Path:
    return MRIO_DIR / f"{TAG}_T-Results_{YEAR}_060_Markup00{m}(full).csv"

def y_results_path(m: int) -> Path:
    return MRIO_DIR / f"{TAG}_Y-Results_{YEAR}_060_Markup00{m}(full).csv"

# ------------------------------------------------------------------
# 1. Load metadata and locate target row
# ------------------------------------------------------------------
regions_df = pd.read_csv(REGIONS_FILE)
sectors_df = pd.read_csv(SECTORS_FILE)

regions = regions_df["Region_acronyms"].tolist()
sectors = sectors_df["sector"].tolist()

r_idx = regions.index(target_region)
s_idx = sectors.index(target_sector)

print("Target region index:", r_idx)
print("Target sector index:", s_idx)

compiled_row_idx = r_idx * P + s_idx
print("Compiled use/fd row index:", compiled_row_idx)

raw_row_idx = r_idx * ROWS_PER_REGION + I + s_idx
print("Raw T/Y row index:", raw_row_idx)

valuation_names = {
    1: "basic prices",
    2: "trade margins",
    3: "transport margins",
    4: "taxes on products",
    5: "subsidies on products"
}

# ------------------------------------------------------------------
# 2. Inspect T (USE) row sums across valuations  (unchanged)
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("T-RESULTS / USE INSPECTION")
print("=" * 80)

t_row_sums = {}
t_row_blocks = {}

for m in range(1, 6):
    fpath = t_results_path(m)
    print("\nReading T:", fpath.name)

    row_df = pd.read_csv(
        fpath,
        header=None,
        skiprows=raw_row_idx,
        nrows=1
    )
    row = row_df.iloc[0]

    vals = []
    for r_col in range(R):
        col_start = r_col * ROWS_PER_REGION
        col_end   = col_start + I
        vals.append(row.iloc[col_start:col_end].to_numpy())

    use_row_m = np.hstack(vals)
    t_row_blocks[m] = use_row_m
    t_row_sums[m] = use_row_m.sum()

    print(f"Valuation {m} ({valuation_names[m]}), USE row sum: {t_row_sums[m]}")

use_row_cp = sum(t_row_blocks[m] for m in range(1, 6))
print("\nCombined purchasers-price USE row sum:", use_row_cp.sum())

# ------------------------------------------------------------------
# 3. Inspect Y (FINAL DEMAND) row sums across valuations
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("Y-RESULTS / FINAL DEMAND INSPECTION")
print("=" * 80)

y_row_sums = {}
y_row_blocks = {}

for m in range(1, 6):
    fpath = y_results_path(m)
    print("\nReading Y:", fpath.name)

    row_df = pd.read_csv(
        fpath,
        header=None,
        skiprows=raw_row_idx,
        nrows=1
    )
    row = row_df.iloc[0]

    vals = []
    for r_col in range(R):
        col_start = r_col * FD
        col_end   = col_start + FD
        vals.append(row.iloc[col_start:col_end].to_numpy())

    fd_row_m = np.hstack(vals)
    y_row_blocks[m] = fd_row_m
    y_row_sums[m] = fd_row_m.sum()

    print(f"Valuation {m} ({valuation_names[m]}), FD row sum: {y_row_sums[m]}")

fd_row_cp = sum(y_row_blocks[m] for m in range(1, 6))
print("\nCombined purchasers-price FD row sum:", fd_row_cp.sum())