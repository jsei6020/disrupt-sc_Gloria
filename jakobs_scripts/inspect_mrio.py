import pickle
import pandas as pd
from pathlib import Path

filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd_cp.pkl"

p = Path(filepath)
if not p.exists():
    print(f"File not found: {p}")
    raise SystemExit(1)

with open(p, 'rb') as f:
    obj = pickle.load(f)

print("Loaded a DataFrame")
print("Shape:", obj.shape)
#print("Columns:", obj.columns.tolist()[:20])
#print("Head:\n", obj.head())
# 5x5 block starting at row 10000, column 10000 (0-based index 9999)
#negative_outputs = [
#        (region_sector, value)
#        for region_sector, value in tot_outputs_per_region_sector.items()
#        if value < 0
#    ]

    # Print the list if there are negatives
#    if negative_outputs:
#        print("⚠️ Found NEGATIVE total outputs in these region-sector pairs:")
row = obj.loc[('XAF', 'Air transport')]

for i in range(19680, 20000):
    label = row.index[i]
    value = row.iloc[i]
    print(f"Region-Sector: {label} | Value: {value}")#    else:
#        print("✅ No negative total outputs found.")
#print(obj.loc[('XAF', 'Air transport')])