def main():
    import pandas as pd
    
    filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/"

    regions = pd.read_csv(filepath + "regions_table.csv", header=None, skiprows=1)[1].tolist()
    sectors = pd.read_csv(filepath + "sector_table_g.csv", header=None, skiprows=1)[2].tolist()
    # Create MultiIndex: each region paired with every sector
    multi_index = pd.MultiIndex.from_product([regions, sectors], names=["region", "sector"])
    print(multi_index)

    df_sparse = None
    df_sparse = mrio.astype(pd.SparseDtype("float", fill_value=0))
    chunk_iter = pd.read_csv(filepath + "mrio_2023.csv", header=None, chunksize=10000)
    for i, chunk in enumerate(pd.read_csv(filepath + "mrio_2023.csv", header=None, chunksize=1312)):
    #    print(i)
        # Convert to sparse
    #    chunk_sparse = chunk.astype(pd.SparseDtype("float", fill_value=0))
        # Incremental concatenation
    #    if df_sparse is None:
    #        df_sparse = chunk_sparse
    #    else:
    #        df_sparse = pd.concat([df_sparse, chunk_sparse])
            #chunk_sparse.to_csv(filepath + "indexed_mrio_2023_" + str(i) + ".csv")
            #print(chunk_sparse)
            #df_sparse.to_csv(filepath + "indexed_mrio_2023.csv")
    #    del chunk, chunk_sparse # delete to help free memory

    df_sparse.index = multi_index

    print(df_sparse)


if __name__ == "__main__":
    main()
