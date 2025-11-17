def main():
    import pandas as pd
    import mario
    import os
    import numpy as np

    filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/"
    pickle_file = filepath + "df_sparse.pkl"


###Create Multi-Index
    regions = pd.read_csv(filepath + "regions_table.csv", header=None, skiprows=1)[1].tolist()
    sectors = pd.read_csv(filepath + "sector_table_g.csv", header=None, skiprows=1)[2].tolist()
    sectors = sectors + sectors
    
    # Create MultiIndex: each region paired with every sector
    multi_index = pd.MultiIndex.from_product([regions, sectors], names=["region", "sector"])
    #print(multi_index)

###Load sparse data from file
    if os.path.exists(pickle_file):
        print("Loading cached sparse DataFrame from disk...")
        df_sparse = pd.read_pickle(pickle_file)
###Load dense data
    else:

        df_sparse = None
        for i, chunk in enumerate(pd.read_csv(filepath + "mrio_2023.csv", header=None, chunksize=1312)):
            print(i)
            # Convert to sparse
            chunk.to_parquet(filepath + "indexed_mrio_2023_" + str(i) + ".parquet")

            chunk_sparse = chunk.astype(pd.SparseDtype("float", fill_value=0))
            # Incremental concatenation
            if df_sparse is None:
                df_sparse = chunk_sparse
            else:
                df_sparse = pd.concat([df_sparse, chunk_sparse])
            del chunk, chunk_sparse # delete to help free memory

        df_sparse.index = multi_index
        df_sparse.to_pickle(filepath + "df_sparse.pkl")





if __name__ == "__main__":
    main()
