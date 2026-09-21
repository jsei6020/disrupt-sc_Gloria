import pandas as pd
from pathlib import Path

# base path
basepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/"

# -------------------------------------------------------------------
# 1. Load matrices
# -------------------------------------------------------------------

use_cp = pd.read_pickle(Path(basepath, "use_cp.pkl"))          # T at purchasers' prices
fd_cp  = pd.read_pickle(Path(basepath, "fd_c_cp.pkl"))         # Y at purchasers' prices
va_b   = pd.read_pickle(Path(basepath, "va_b.pkl"))            # V at basic prices

print("use_cp shape:", use_cp.shape)
print("va_b  shape:", va_b.shape)
print("fd_cp shape:", fd_cp.shape)

# -------------------------------------------------------------------
# 2. Stack use and value added vertically
# -------------------------------------------------------------------

# Ensure VA columns align with USE columns (they should from your build)
va_b = va_b.reindex(columns=use_cp.columns)

use_va_cp = pd.concat([use_cp, va_b], axis=0)

print("use_va_cp shape:", use_va_cp.shape)

# -------------------------------------------------------------------
# 3. Append final demand to the right
# -------------------------------------------------------------------

# Make sure FD index aligns with use+VA rows (industries + VA rows)
fd_cp = fd_cp.reindex(index=use_va_cp.index)

mrio_va_fd_cp = pd.concat([use_va_cp, fd_cp], axis=1)

print("mrio_va_fd_cp shape:", mrio_va_fd_cp.shape)

# -------------------------------------------------------------------
# 4. Save final MRIO (consumer-price for flows, basic-price VA)
# -------------------------------------------------------------------

out_pkl = Path(basepath, "mrio_va_fd_cp.pkl")
#out_csv = Path(basepath, "mrio_va_fd_cp.csv")

mrio_va_fd_cp.to_pickle(out_pkl)
#mrio_va_fd_cp.to_csv(out_csv)

print("Saved MRIO with VA+FD at mixed valuation (flows at consumer prices, VA at basic):")
print(" ", out_pkl)
#print(" ", out_csv)