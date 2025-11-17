import pickle
import numpy as np
from scipy import sparse
import pandas as pd
import os
import numpy as np
import gc


def load_dataframe_from_pickle(path):
    
    ###Load sparse data from file
    if os.path.exists(path):
        print("Loading cached sparse DataFrame from disk...")
        df_sparse = pd.read_pickle(path)
        return df_sparse
    

def extract_product_use_blocks(mrsut_df, R, I, P, multi_index):
    """
    Extract the (products_block, products_block) from each (region, region) combination.
    Returns a DataFrame with shape (R*P, R*P).
    """
    # Preallocate output DataFrame (or use sparse DataFrame for memory)
    all_region_product_idx = []
    blocks = []
    for r_row in range(R):
        row_start = r_row * (I + P) + I
        row_end   = row_start + P
        row_idx = mrsut_df.index[row_start:row_end]
        row_blocks = []
        print("Start: " + str(row_start))
        for r_col in range(R):
            col_start = r_col * (I + P)
            col_end   = col_start + P
            col_idx = mrsut_df.columns[col_start:col_end]
            block = mrsut_df.iloc[row_start:row_end, col_start:col_end]
            row_blocks.append(block.values)
        blocks.append(np.hstack(row_blocks))
        all_region_product_idx += list(row_idx)
    #blocks.to_pickle(filepath + "blocks_sparse.pkl")
    #print("Blocks Saved")
    del mrsut_df
    gc.collect()

    iot_matrix = np.vstack(blocks)
    #iot_matrix.to_pickle(filepath + "iotm_sparse.pkl")
    #print("IOT matrix Saved")
    del blocks
    gc.collect()
    print("Trash collected")

    # Recompose a DataFrame for output
    return pd.DataFrame(iot_matrix, index=multi_index, columns=multi_index)

def main():
    filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/"
    regions = pd.read_csv(filepath + "regions_table.csv", header=None, skiprows=1)[1].tolist()
    sectors = pd.read_csv(filepath + "sector_table_g.csv", header=None, skiprows=1)[2].tolist()
    # Create MultiIndex: each region paired with every sector
    multi_index = pd.MultiIndex.from_product([regions, sectors], names=["region", "sector"])
    print(multi_index)
    del regions,sectors
    gc.collect()

    pickle_file = filepath + "df_sparse.pkl"    
    R = 164
    I = 120
    P = 120
    mrsut_df = load_dataframe_from_pickle(pickle_file)
    print("Loaded MR-SUT. Shape:", mrsut_df.shape)
    iot = extract_product_use_blocks(mrsut_df, R, I, P, multi_index)
    print("Extracted IOT shape:", iot.shape)
    iot.to_pickle(filepath + "iot_sparse.pkl")

    iot.to_csv(filepath + "GLORIA_IOT.csv")
    print("Saved as GLORIA_IOT.csv")

if __name__ == "__main__":
    main()
