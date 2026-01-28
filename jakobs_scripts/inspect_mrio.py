import pickle
import pandas as pd
from pathlib import Path

filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/mrio_va_fd59.pkl"

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
print(obj.iloc[2000:2006, 2000:2006])