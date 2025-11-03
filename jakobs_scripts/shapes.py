import pandas as pd

df = pd.read_csv("/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global2/Economic/mrio_b.csv", index_col=[0,1], header=[0,1])
print(df.shape)

rows = set(df.index)
cols = set(df.columns)

missing_rows = cols - rows      # appear in columns but not rows
missing_cols = rows - cols      # appear in rows but not columns

print("Missing rows:", len(missing_rows))
print("Missing columns:", len(missing_cols))

print("Examples of missing rows:", list(missing_rows)[:5])
print("Examples of missing columns:", list(missing_cols)[:5])

# Normalize and check duplicates
df.index = pd.MultiIndex.from_tuples(
    [(r.lower().strip(), s.lower().strip()) for r, s in df.index]
)
df.columns = pd.MultiIndex.from_tuples(
    [(r.lower().strip(), s.lower().strip()) for r, s in df.columns]
)

dupes_rows = df.index.duplicated().sum()
dupes_cols = df.columns.duplicated().sum()
print("Duplicate rows:", dupes_rows)
print("Duplicate columns:", dupes_cols)


#all_labels = sorted(rows | cols)
#df_square = df.reindex(index=all_labels, columns=all_labels, fill_value=0)
#print(df_square.shape)

rows = set(df.index)
cols = set(df.columns)
all_labels = sorted(rows | cols)

df_square = df.reindex(index=all_labels, columns=all_labels, fill_value=0)
print(df_square.shape)


#df_square.to_csv("/home/user/Documents/University/Master Thesis/disrupt-sc/data/Global2/Economic/mrio_b.csv.csv")


###analysis code        
        df_cols = pd.DataFrame(intermediate_cols, columns=['Country', 'Sector'])
        print(df_cols)
        df_cols.to_csv("/home/user/Downloads/cols.csv")
        
        df_rows = pd.DataFrame(intermediate_rows, columns=['Country', 'Sector'])
        df_rows.to_csv("/home/user/Downloads/rows.csv")
###end analysis 


###Addition for the global table
        row_exports_pair = ('ROW', 'Exports')
        # Check if the ROW Exports column exists
        if row_exports_pair in self.columns:
            # Normal row sum across all *existing columns except ROW Exports*
            filtered = [pair for pair in selected_industries if pair != row_exports_pair]
            base_sum = self.loc[filtered].sum(axis=1)
            # Add the ROW Exports column values (vector) element-wise
            total_output = base_sum.add(self[row_exports_pair], fill_value=0)
            return total_output
            #return base_sum
        else:
            # Fallback: no ROW Exports column present, regular output sum

###Country modeling for Global Scope
        #from disruptsc.main import parse_arguments
        #args_general=parse_arguments()
        #logging.info(args_general.scope)
        #if args_general.scope == "Global1":
        #    self.external_buying_countries = ["ROW"]
        #    self.external_selling_countries = ["ROW"]
        #    print(self.external_buying_countries)
        #    print(self.external_selling_countries)
        #else:

        ###Addition for the global table
        #row_exports_pair = ('ROW', 'Exports')
        # Check if the ROW Exports column exists
        #if row_exports_pair in self.columns:
            # Normal row sum across all *existing columns except ROW Exports*
        #    self.region_sectors = [pair for pair in self.region_sectors if pair != row_exports_pair]
        import networkx as nx
            print(transport_network.nodes[9789])
            print(transport_network.nodes[4297])
            print(nx.has_path(transport_network, 9789, 4297))
            print("yay####################################")

            components = list(nx.connected_components(transport_network))
            print(f"Number of connected components: {len(components)}")

            # Identify which component 9789 and 4297 belong to
            for i, comp in enumerate(components):
                if 9789 in comp or 4297 in comp:
                    print(f"Node 9789 in component {i}: {9789 in comp}")
                    print(f"Node 4297 in component {i}: {4297 in comp}")