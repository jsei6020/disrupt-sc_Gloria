import pandas as pd
filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/"
mrio_va = pd.read_pickle(filepath+"mrio_va.pkl")
fd = pd.read_pickle(filepath+"fd_c.pkl")

result = pd.concat([mrio_va, fd], axis=1)
result.to_pickle(filepath + "mrio_va_fd.pkl")
