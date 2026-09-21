import geopandas as gpd
import matplotlib.pyplot as plt

# Load transport flows
flows = gpd.read_file('/home/user/Documents/University/Master Thesis/disrupt-sc/output/Global2/20251023_150309/transport_edges_with_flows_0.geojson')

# Plot flow intensity
flows.plot(column='flow_total', linewidth=2, cmap='Reds')
plt.title('Transport Flow Intensity')
plt.show()