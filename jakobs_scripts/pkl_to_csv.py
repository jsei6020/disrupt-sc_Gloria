import pickle
import pandas as pd
import numpy as np

filepath = "/home/user/Documents/University/Master Thesis/Databases/Gloria/IO-Table/"

with open(filepath + "supply.pkl", "rb") as f:
    data = pickle.load(f)      # this is a numpy array
#import numpy as np

# data is your 2D numpy array
#coords = np.argwhere(data != 0)

#if coords.size == 0:
#    print("All elements are zero")
#else:
#    r, c = coords[0]
#    print(f"First non‑zero at row {r}, column {c}, value = {data[r, c]}")
#df = pd.DataFrame(data)        # wrap into DataFrame
#diag = np.diag(data)   # or np.diag(df.values)
print(df)

df.to_csv(filepath + "use.csv", index=False)