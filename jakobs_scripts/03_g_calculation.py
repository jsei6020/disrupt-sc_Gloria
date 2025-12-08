import pandas as pd
import numpy as np

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/"
use = pd.read_pickle(filepath+"use.pkl")
va = pd.read_pickle(filepath+"va_c.pkl")

# Align columns
#va = va.reindex(columns=use.columns)

# Append VA below use
use_va = pd.concat([use, va], axis=0)
print(use_va)
# Column sums → industry output vector g
g = use_va.sum(axis=0)            # Series
print(g)
# Diagonal matrix as DataFrame
g_array = g.to_numpy()
G_df = pd.DataFrame(
    np.diag(g_array),
    index=use.columns,
    columns=use.columns
)

# Export vector and diagonal matrix
g.to_csv(filepath + "g_vector.csv")

G_df.to_pickle(filepath + "G_diag.pkl")
G_df.to_csv(filepath + "G_diag.csv")


#mrio = pd.read_pickle(filepath+"use.pkl")
#fd = pd.read_pickle(filepath+"fd_c.pkl")

#result = pd.concat([mrio_va, fd], axis=1)
#result.to_pickle(filepath + "mrio_va_fd.pkl")