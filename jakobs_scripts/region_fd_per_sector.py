import pandas as pd

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/"
df = pd.read_pickle(filepath + "fd_c.pkl")  # MultiIndex rows & columns [web:14][web:22]

# the six categories (second level of column MultiIndex)
fd_cats = [
    "Household final consumption P.3h",
    "Non-profit institutions serving households P.3n",
    "Government final consumption P.3g",
    "Gross fixed capital formation P.51",
    "Changes in inventories P.52",
    "Acquisitions less disposals of valuables P.53",
]

# select all columns whose category is in fd_cats
col_regions = df.columns.get_level_values(0)
col_cats    = df.columns.get_level_values(1)

fd_cols = df.loc[:, col_cats.isin(fd_cats)]

# sum over the six categories *per column-region* for each (row-region, sector)
# this collapses the category level but keeps column-region separate
# then sum across column-regions to get total final demand per (row-region, sector)
tmp = fd_cols.groupby(level=0, axis=1).sum()   # sum over categories for each column-region [web:9]
row_total = tmp.sum(axis=1)                    # sum over all column-regions

# row index is already (region, sector)
result_df = row_total.to_frame(name="total_final_demand")

# export
outpath = filepath + "region_sector_total_final_demand.csv"
result_df.to_csv(outpath)
