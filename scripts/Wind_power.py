import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import atlite
from atlite.gis import ExclusionContainer
import xarray as xr
from shapely.ops import unary_union
import networkx as nx
import numpy as np
import yaml


with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Snakemake parameter for selected system and max distance
selected_system_id = config['scenario']['selected_system_id']
#selected_system_id = snakemake.params.system_id
year_of_interest = config['scenario']['year_of_interest']
#year_of_interest = snakemake.params.year_of_interest

start_date = f'{year_of_interest}-01-01'
end_date = f'{year_of_interest}-12-31 23:00'

timestamps = pd.date_range(start=start_date, end=end_date, freq='h')

electricity_market_file = f"data/electricity_price/energy-charts_Stromproduktion_und_Boersenstrompreise_in_Deutschland_{year_of_interest}.csv"
electricity_market = pd.read_csv(electricity_market_file)

electricity_market.index = timestamps
electricity_market = electricity_market.drop('Datum (MEZ)', axis=1)

electricity_market = electricity_market['Day Ahead Auktion Preis (EUR/MWh; EUR/tCO2)']

pa = gpd.read_file('../data/RLI-potentialareas_wind_pv_v1.0/data/potentialarea_wind_settlement-1000m.gpkg')
pa = pa.set_index("id").drop(['country_id'], axis=1)
pa = pa.to_crs(3035)
area_wind = pa.area / 1e6 # kann wahrscheinlich weg
area_wind.name = "area_wind" # kann wahrscheinlich weg
pa_new = pd.concat([pa, area_wind], axis=1)


url = "https://tubcloud.tu-berlin.de/s/RHZJrN8Dnfn26nr/download/NUTS_RG_10M_2021_4326.geojson"
nuts = gpd.read_file(url)
nuts = nuts.set_index("id")
nuts1 = nuts.query("LEVL_CODE == 1").to_crs(3035)
subregion = nuts1.query("CNTR_CODE == 'DE'")

joined = pa_new.sjoin(subregion)
if 'index_right' in joined.columns:
    joined.rename(columns={'index_right': 'joined_index_right'}, inplace=True)

"""
Reading shapefile Finsterwalde district heating Netz
"""

selected_system = gpd.read_file(f'output/selected_system_{selected_system_id}.gpkg')
selected_system = selected_system.to_crs("EPSG:3035")

# Example: Spatial join with a maximum distance of 'y' meters
max_distance = 10e3  # Set 'y' to your desired maximum distance
close_areas = gpd.sjoin_nearest(joined, selected_system, max_distance=max_distance, distance_col='distance')


#close_areas['area'] = close_areas.geometry.area
min_area = 0.5  # Mindestfläche in Quadratmetern km
close_areas = close_areas[close_areas['area_wind'] >= min_area]


buffer_size = 10  # Example buffer size in meters; adjust based on your requirements
close_areas['buffered_geometry'] = close_areas.geometry.buffer(buffer_size)


# Create a single MultiPolygon from the buffered geometries
G = nx.Graph()

for index, row in close_areas.iterrows():
    G.add_node(index, geometry=row['buffered_geometry'])

# Add an edge between polygons that intersect
for index1, geom1 in G.nodes(data='geometry'):
    for index2, geom2 in G.nodes(data='geometry'):
        if index1 != index2 and geom1.intersects(geom2):
            G.add_edge(index1, index2)

# Find connected components - these are your groups
groups = list(nx.connected_components(G))

new_geometries = [unary_union([G.nodes[i]['geometry'] for i in group]) for group in groups]

combined_areas = gpd.GeoDataFrame({'geometry': new_geometries}, crs=close_areas.crs)

combined_areas.plot()
plt.show()

max_area = close_areas['area_wind'].max()
row_max_area = close_areas[close_areas['area_wind'] == max_area]
max_area_geometry = row_max_area['geometry']
max_area_geometry.to_file(f"output/wind_potentials_{selected_system_id}.gpkg", driver="GPKG")

"""
Hier neuer Abschnitt 
"""

url = "https://tubcloud.tu-berlin.de/s/7bpHrAkjMT3ADSr/download/country_shapes.geojson"
countries = gpd.read_file(url).set_index("name")

excluder = ExclusionContainer(crs=3035)

cutout = atlite.Cutout(f"input/cutout_germany/germany_{year_of_interest}.nc")

A = cutout.availabilitymatrix(combined_areas, excluder)

cap_per_sqkm = 7.3 # 2 MW/km^2
area = cutout.grid.set_index(["y", "x"]).to_crs(3035).area / 1e6 # in km^2
area = xr.DataArray(area, dims=("spatial"))

capacity_matrix = A.stack(spatial=["y", "x"]) * area * cap_per_sqkm

wind = cutout.wind(matrix=capacity_matrix, turbine="Vestas_V90_3MW", add_cutout_windspeed=True, index=combined_areas.index)

wind_speed = cutout.data.wnd100m

geometry = selected_system.geometry.iloc[0]

centroid = geometry.centroid
x_point, y_point = centroid.x, centroid.y


wind_speed_at_point = wind_speed.sel(x=x_point, y=y_point, method='nearest')
wind_speed_at_point = wind_speed_at_point.to_series()


print(wind.dims)

wind.isel(time=0).plot()
plt.show()

total_wind_power = wind.sum(dim='time')
max_power_index = total_wind_power.argmax(dim='dim_0')
max_power_series = wind.isel(dim_0=max_power_index)
wind_series = max_power_series.to_series()

redispatch = electricity_market <= 0

negative_values_count = (electricity_market <= 0).sum()
print("Number of values less than 0:", negative_values_count)

redispatch_wind_series = np.where(redispatch, wind_series, 0)

redispatch_wind_series = pd.Series(redispatch_wind_series, index=wind_series.index)
redispatch_wind_series.to_csv(f'output/redispatch_wind_series_{selected_system_id}_{year_of_interest}')
