import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd

import cartopy.crs as ccrs

import atlite
from atlite.gis import ExclusionContainer

import xarray as xr

import fiona
from shapely.geometry import MultiLineString


pa = gpd.read_file('data/RLI-potentialareas_wind_pv_v1.0/data/potentialarea_wind_settlement-1000m.gpkg')
pa = pa.set_index("id").drop(['country_id'], axis=1)
pa = pa.to_crs(3035)
area_wind = (pa.area / 1e6)
area_wind.name = "area_wind"
pa_new = pd.concat([pa, area_wind], axis=1)
print(pa.crs)
print(pa.total_bounds)

url = "https://tubcloud.tu-berlin.de/s/RHZJrN8Dnfn26nr/download/NUTS_RG_10M_2021_4326.geojson"
nuts = gpd.read_file(url)
print(nuts.head(3))
nuts = nuts.set_index("id")
print(nuts.geometry)
print(nuts.crs)
nuts1 = nuts.query("LEVL_CODE == 1").to_crs(3035)
subregion = nuts1.query("CNTR_CODE == 'DE'")

joined = pa_new.sjoin(subregion)
if 'index_right' in joined.columns:
    joined.rename(columns={'index_right': 'joined_index_right'}, inplace=True)

fig = plt.figure(figsize=(7, 7))

ax = plt.axes(projection=ccrs.epsg(3035))

subregion.plot(ax=ax, edgecolor="black", facecolor="lightgrey")

joined.plot(
    ax=ax, column="area_wind", #markersize=joined.area_wind, legend=True
)

ax.set_extent([5, 19, 47, 55])

fig.show()

"""
Reading shapefile Finsterwalde district heating Netz
"""

with fiona.Env(SHAPE_RESTORE_SHX='YES'):
    FiWa_Netz = gpd.read_file('data/Waermenetz_Finsterwalde/FiWa_Bestandsfernwärmenetz.shp')

print("Original CRS:", FiWa_Netz.crs)

# Combine geometries to a MultiLineString
multi_line = FiWa_Netz.geometry.unary_union

# Check if the resulting geometry is a MultiLineString
if isinstance(multi_line, MultiLineString):
    print("Successfully combined into a MultiLineString.")
else:
    print("Result is not a MultiLineString. It's a", type(multi_line))

# Create a new GeoDataFrame with the MultiLineString
FiWa_Netz = gpd.GeoDataFrame(geometry=[multi_line])

# Set the CRS for the GeoDataFrame (replace 'original_epsg' with the actual EPSG code of your original data)
FiWa_Netz.crs = 'EPSG:25833'

# Reproject to EPSG:3035
FiWa_Netz = FiWa_Netz.to_crs(epsg=3035)

# Now you can plot the GeoDataFrame
#FiWa_Netz.plot(color='blue', linewidth=2)
#plt.show()

# Example: Spatial join with a maximum distance of 'y' meters
max_distance = 10e3  # Set 'y' to your desired maximum distance
close_areas = gpd.sjoin_nearest(joined, FiWa_Netz, max_distance=max_distance, distance_col='distance')

fig, ax = plt.subplots(figsize=(10, 10))  # You can adjust the size as needed

# Plot FiWa_Netz
FiWa_Netz.plot(ax=ax, color='blue', linewidth=2, label='FiWa_Netz')

# Plot 'joined'
close_areas.plot(ax=ax, color='red', alpha=0.5, label='Joined Areas')

# Set plot title and labels
ax.set_title('FiWa_Netz and Joined Areas')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add a legend
ax.legend()

# Show the plot
plt.show()
"""
Hier neuer Abschnitt 
"""

url = "https://tubcloud.tu-berlin.de/s/7bpHrAkjMT3ADSr/download/country_shapes.geojson"
countries = gpd.read_file(url).set_index("name")

excluder = ExclusionContainer(crs=3035)

cutout = atlite.Cutout("Germany-2019.nc")

A = cutout.availabilitymatrix(close_areas, excluder)

cap_per_sqkm = 2 # 2 MW/km^2
area = cutout.grid.set_index(["y", "x"]).to_crs(3035).area / 1e6 # in km^2
area = xr.DataArray(area, dims=("spatial"))

capacity_matrix = A.stack(spatial=["y", "x"]) * area * cap_per_sqkm

cutout.prepare()
wind = cutout.wind(matrix=capacity_matrix, turbine="Vestas_V90_3MW", index=close_areas.index)

wind.isel(time=0).plot()
plt.show()

