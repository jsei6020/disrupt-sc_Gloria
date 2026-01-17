import pandas as pd
import numpy as np

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/"
W = filepath + "GLORIA_MRIOs_59_2022/20240419_120secMother_AllCountries_002_V-Results_2022_059_Markup001(full).csv"
Y = filepath + "GLORIA_MRIOs_59_2022/20240110_120secMother_AllCountries_002_Y-Results_2022_059_Markup001(full).csv"

REGIONS_FILE = filepath + "IO-Table/regions_table.csv"
SECTORS_FILE = filepath + "IO-Table/sector_table_g.csv"
regions = pd.read_csv(REGIONS_FILE, header=None, skiprows=1)[1].tolist()
sectors = pd.read_csv(SECTORS_FILE, header=None, skiprows=1)[2].tolist()
# Create MultiIndex: each region paired with every sector
multi_index = pd.MultiIndex.from_product([regions, sectors])
    
fd = pd.read_csv(Y, header=None)
print(fd.shape)
fd_labels = pd.read_csv(filepath + "IO-Table/fd_names.csv", header=None)[0].tolist()
print(fd_labels)
multi_index2 = pd.MultiIndex.from_product([regions, fd_labels])

va = pd.read_csv(W, header=None)
print(va.shape)
va_labels = pd.read_csv(filepath + "IO-Table/va_names.csv", header=None)[0].tolist()
print(va_labels)
multi_index3 = pd.MultiIndex.from_product([regions, va_labels])

blocks = []

R = 164
I = 120
P = 120
FD = 6
VA = 6

#final demand
for r_row in range(R):
    row_start = r_row * (I+P) + I
    row_end   = row_start + P
    row_blocks = []
    print("Start: " + str(row_start) + " End:" + str(row_end))
    for r_col in range(R):
        col_start = r_col * (FD)
        col_end   = col_start + FD
        block = fd.iloc[row_start:row_end, col_start:col_end]
        row_blocks.append(block.values)
    blocks.append(np.hstack(row_blocks))
    del row_blocks

del fd
fd_matrix = np.vstack(blocks)
del blocks

# Recompose a DataFrame for output
fd_c = pd.DataFrame(fd_matrix, index=multi_index, columns=multi_index2)
fd_c.to_pickle(filepath + "IO-Table/fd_c.pkl")
fd_c.to_csv(filepath + "IO-Table/fd_c.csv")

#value added
blocks = []

for r_row in range(R):
    row_start = r_row * (VA)
    row_end   = row_start + VA
    row_blocks = []
    print("Start: " + str(row_start) + " End:" + str(row_end))
    for r_col in range(R):
        col_start = r_col * (I+P)
        col_end   = col_start + I
        block = va.iloc[row_start:row_end, col_start:col_end]
        row_blocks.append(block.values)
    blocks.append(np.hstack(row_blocks))
    del row_blocks

del va
va_matrix = np.vstack(blocks)
del blocks

# Recompose a DataFrame for output
va_c = pd.DataFrame(va_matrix, index=multi_index3, columns=multi_index)
va_c.to_pickle(filepath + "IO-Table/va_c.pkl")
va_c.to_csv(filepath + "IO-Table/va_c.csv")

