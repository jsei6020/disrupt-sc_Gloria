import pandas as pd
import numpy as np

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/"

use = pd.read_pickle(filepath + "use.pkl")
va  = pd.read_pickle(filepath + "va_c.pkl")
fd  = pd.read_pickle(filepath + "fd_c.pkl")

# 1. Append VA below use
# Make sure VA has same columns as use
#va = va.reindex(columns=use.columns)
use_va = pd.concat([use, va], axis=0)

# 2. Append final demand to the right
# Make sure fd has same index as use_va (typically industries + VA rows if needed)
# If fd is only for industries, align on index:
#fd = fd.reindex(index=use_va.index)
mrio_va_fd = pd.concat([use_va, fd], axis=1)
print(mrio_va_fd)
# Export combined table
mrio_va_fd.to_pickle(filepath + "mrio_va_fd.pkl")
#mrio_va_fd.to_csv(filepath + "mrio_va_fd.csv")
