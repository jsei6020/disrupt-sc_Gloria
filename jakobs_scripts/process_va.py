import pandas as pd
import numpy as np

filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/"
regions = pd.read_csv(filepath + "regions_table.csv", header=None, skiprows=1)[1].tolist()
sectors = pd.read_csv(filepath + "sector_table_g.csv", header=None, skiprows=1)[2].tolist()
# Create MultiIndex: each region paired with every sector
multi_index = pd.MultiIndex.from_product([regions, sectors])
    
fd = pd.read_csv(filepath + "final_demand_2023.csv", header=None)
print(fd.shape)

fd_labels = pd.read_csv(filepath + "final_demand_names.csv", header=None)[0].tolist()
print(fd_labels)
multi_index2 = pd.MultiIndex.from_product([regions, fd_labels])


blocks = []

R = 164
I = 120
P = 120
F = 6

for r_row in range(R):
    row_start = r_row * (I+P) + I
    row_end   = row_start + P
    row_blocks = []
    print("Start: " + str(row_start) + " End:" + str(row_end))
    for r_col in range(R):
        col_start = r_col * (F)
        col_end   = col_start + F
        block = fd.iloc[row_start:row_end, col_start:col_end]
        row_blocks.append(block.values)
    blocks.append(np.hstack(row_blocks))
    del row_blocks

del fd
fd_matrix = np.vstack(blocks)
del blocks




# Recompose a DataFrame for output
fd_c = pd.DataFrame(fd_matrix, index=multi_index, columns=multi_index2)
fd_c.to_pickle(filepath + "fd_c.pkl")
fd_c.to_csv(filepath + "fd_c.csv")


