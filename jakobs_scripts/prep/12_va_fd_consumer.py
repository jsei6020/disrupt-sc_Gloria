import pandas as pd
import numpy as np

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/"

# --------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------
REGIONS_FILE = filepath + "IO-Table/regions_table.csv"
SECTORS_FILE = filepath + "IO-Table/sector_table_g.csv"

regions = pd.read_csv(REGIONS_FILE, header=None, skiprows=1)[1].tolist()
sectors = pd.read_csv(SECTORS_FILE, header=None, skiprows=1)[2].tolist()
multi_index = pd.MultiIndex.from_product([regions, sectors])

fd_labels = pd.read_csv(filepath + "IO-Table/fd_names.csv", header=None)[0].tolist()
multi_index2 = pd.MultiIndex.from_product([regions, fd_labels])

va_labels = pd.read_csv(filepath + "IO-Table/va_names.csv", header=None)[0].tolist()
multi_index3 = pd.MultiIndex.from_product([regions, va_labels])

R = 164
I = 120
P = 120
FD = 6
VA = 6

# --------------------------------------------------------------------
# 1) FINAL DEMAND at consumer prices: sum over valuations, then slice
# --------------------------------------------------------------------

# sum the five Y-Results sheets
fd_sum = None
for m in range(1, 6):
    Ym = (
        filepath
        + f"GLORIA_MRIOs_60_2023/"
          f"20251217_120secMother_AllCountries_002_Y-Results_2023_060_Markup00{m}(full).csv"
    )
    print("Reading FD valuation:", Ym)
    fd_m = pd.read_csv(Ym, header=None)
    if fd_sum is None:
        fd_sum = fd_m.astype(np.float64)
    else:
        fd_sum += fd_m
    del fd_m

# your original block logic, now applied to fd_sum
blocks = []

for r_row in range(R):
    row_start = r_row * (I + P) + I
    row_end   = row_start + P
    row_blocks = []
    print("FD rows Start:", row_start, "End:", row_end)
    for r_col in range(R):
        col_start = r_col * FD
        col_end   = col_start + FD
        block = fd_sum.iloc[row_start:row_end, col_start:col_end]
        row_blocks.append(block.values)
    blocks.append(np.hstack(row_blocks))
    del row_blocks

del fd_sum
fd_matrix_cp = np.vstack(blocks)
del blocks

fd_c_cp = pd.DataFrame(fd_matrix_cp, index=multi_index, columns=multi_index2)
fd_c_cp.to_pickle(filepath + "IO-Table/fd_c_cp.pkl")
fd_c_cp.to_csv(filepath + "IO-Table/fd_c_cp.csv")

# --------------------------------------------------------------------
# 2) VALUE ADDED at basic prices: single valuation, original logic
# --------------------------------------------------------------------

W = (
    filepath
    + "GLORIA_MRIOs_60_2023/"
      "20251217_120secMother_AllCountries_002_V-Results_2023_060_Markup001(full).csv"
)
print("Reading VA (basic prices):", W)
va = pd.read_csv(W, header=None)

blocks = []

for r_row in range(R):
    row_start = r_row * VA
    row_end   = row_start + VA
    row_blocks = []
    print("VA rows Start:", row_start, "End:", row_end)
    for r_col in range(R):
        col_start = r_col * (I + P)
        col_end   = col_start + I
        block = va.iloc[row_start:row_end, col_start:col_end]
        row_blocks.append(block.values)
    blocks.append(np.hstack(row_blocks))
    del row_blocks

del va
va_matrix = np.vstack(blocks)
del blocks

va_c = pd.DataFrame(va_matrix, index=multi_index3, columns=multi_index)
va_c.to_pickle(filepath + "IO-Table/va_b.pkl")
va_c.to_csv(filepath + "IO-Table/va_b.csv")