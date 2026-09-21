import pandas as pd
from pathlib import Path

base_path = Path("/home/user/Documents/University/Master Thesis/vulnerability_analysis")

v_is_path = base_path / "v-is.csv"
final_demand_path = base_path / "total_region_final_demand.csv"
output_path = base_path / "v-is_fd.csv"

v_is = pd.read_csv(v_is_path)
final_demand = pd.read_csv(final_demand_path)

final_demand = final_demand[["iso", "total_final_demand"]]

merged = v_is.merge(final_demand, on="iso", how="left")

merged["intensity_score"] = merged["total_loss"] / merged["total_final_demand"] /  760

result = merged[["iso", "region_name", "total_loss", "total_final_demand", "intensity_score"]]
result = result.sort_values('intensity_score', ascending=False).reset_index(drop=True)

result.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(result.head())