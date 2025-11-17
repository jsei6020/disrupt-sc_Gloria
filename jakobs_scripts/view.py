import pandas as pd
filepath = "/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global4/Network/"
mrio_va_fd = pd.read_pickle(filepath+"mrio_va_fd.pkl")
print(mrio_va_fd.shape)
#print(mrio_va_fd)

selected_industries = mrio_va_fd.columns #[tup for tup in iot.columns]  
selected_industries1 = mrio_va_fd.index
input = mrio_va_fd[selected_industries].sum()

import numpy as np
chunk_size = 1000
n = mrio_va_fd.shape[0]
row_sums = []

for i in range(0, n, chunk_size):
    chunk = mrio_va_fd.iloc[i:i+chunk_size]
    row_sums.append(chunk.sum(axis=1).values)

row_sums = np.concatenate(row_sums)
#row_sums.index = mrio_va_fd.index
output = pd.Series(row_sums, index=mrio_va_fd.index)

def detect_level1_label(pattern: str, axis: int):
        sectors = pd.Series([], dtype=str)
        if axis == 0:
            sectors = mrio_va_fd.index.get_level_values(1)
        elif axis == 1:
            sectors = mrio_va_fd.columns.get_level_values(1)
        else:
            ValueError("Wrong axis selected")
        labels = sectors[sectors.str.contains(pattern, case=False)]
        if len(labels) == 0:
            #logging.warning(f"Failed to detect the label used for {pattern} in the MRIO")
            return ""
        else:
            labels = labels.unique()
            labels = labels.values
            return labels 

#output = mrio_va_fd.loc[selected_industries1].sum(axis=1)
export_label = detect_level1_label('export', axis=1)
final_demand_label = detect_level1_label('final.?demand|P.3|P.52|P.53', axis=1) #P.51| Capital
capital_label = detect_level1_label('capital', axis=1)
import_label = detect_level1_label('import', axis=0)
value_added_label = detect_level1_label('value.?added|va|D.1|D.39|B.2n|B.3n|K.1', axis=0) #D.29 in tax?
tax_label = detect_level1_label('tax', axis=0)     

region_sectors1 = [tup for tup in mrio_va_fd.columns
                               if tup[1] not in list(final_demand_label) + list(export_label) + list(capital_label)]
region_sectors2 = [tup for tup in mrio_va_fd.index
                               if tup[1] not in list(value_added_label) + list(import_label) + list(tax_label)]
        
print(input)
print(output)

print(input[region_sectors1] - output[region_sectors2])


