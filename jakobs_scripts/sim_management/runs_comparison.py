import pandas as pd
from itertools import zip_longest

# 1. Load both CSV files
df_top3 = pd.read_csv('/home/user/Documents/University/Master Thesis/supply_concentrations/top3_filtered_export_sectors_per_region.csv')
df_selected = pd.read_csv('/home/user/Documents/University/Master Thesis/supply_concentrations/region_sector_pairs_selected.csv')

# 2. Extract columns, drop NaNs, strip whitespace, and convert to sets
top3_set = set(df_top3['iso_region_sector'].dropna().str.strip())
selected_set = set(df_selected['region_sector'].dropna().str.strip())

# 3. Perform set comparisons
both = sorted(list(top3_set.intersection(selected_set)))
only_top3 = sorted(list(top3_set.difference(selected_set)))
only_selected = sorted(list(selected_set.difference(top3_set)))

# 4. Align lists to the same length for tabular output
rows = list(zip_longest(both, only_top3, only_selected, fillvalue=''))

# 5. Create DataFrame and export to CSV
df_result = pd.DataFrame(rows, columns=[
    'Both_Files', 
    'Only_Top3_Exports', 
    'Only_Region_Sector_Pairs'
])
df_result.to_csv('region_sector_comparison.csv', index=False)

print("✅ Comparison complete! Saved to 'region_sector_comparison.csv'")
print(f"  • Present in both files: {len(both)}")
print(f"  • Only in top3 exports:  {len(only_top3)}")
print(f"  • Only in selected list: {len(only_selected)}")