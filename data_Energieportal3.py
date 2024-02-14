from owslib.wfs import WebFeatureService
import geopandas as gpd
import pandas as pd
import io
import snakemake
import matplotlib.pyplot as plt


# Function to fetch data from the WFS service and load into a GeoDataFrame
def fetch_layer(wfs, layer_name):
    response = wfs.getfeature(typename=layer_name, outputFormat='json')
    data = io.BytesIO(response.read())
    return gpd.read_file(data)


# Function to reproject GeoDataFrame to a given CRS
def reproject_to_crs(gdf, crs):
    return gdf.to_crs(crs)


# Function to find close potentials within a specified distance
def find_close_potentials(potential_gdf, system_geom, distance):
    search_area = system_geom.buffer(distance)
    close_potentials = potential_gdf[potential_gdf.intersects(search_area)]
    return close_potentials


# URL of the WFS service
url = 'https://energieportal-brandenburg.de/geoserver/bepo/ows'

# Connect to the WFS
wfs = WebFeatureService(url=url, version='2.0.0')

# Layers to be used
layer_keys = [
    #'bepo:Gemarkungen_Abwasser',
    #'bepo:Gemarkungen_Abwaerme_Industrie',
    #'bepo:Eignung_EWK',
    #'bepo:Fläche_EWK',
    'bepo:Brandenburg_Fernwaerme',
    'bepo:Gemarkungen_Flussthermie',
    'bepo:Gemarkungen_Seethermie',
    #'bepo:datenquellen',
    #'bepo:Ausschuss'
]

# Fetch and store layers in a dictionary
layer_data = {key: fetch_layer(wfs, key) for key in layer_keys}

# Reproject layers to the specified CRS (EPSG:3035)
for key in layer_data:
    layer_data[key] = reproject_to_crs(layer_data[key], 'EPSG:3035')

# Snakemake parameters for selected system and max distance
selected_system_id = 'Brandenburg_Fernwaerme.3'  # snakemake.config['selected_system_id']
max_distance = 1000    # snakemake.config['max_distance']

# Select the specific district heating system
fernwaerme_df = layer_data['bepo:Brandenburg_Fernwaerme']
selected_system = fernwaerme_df[fernwaerme_df['id'] == selected_system_id]
selected_system_geom = selected_system.geometry.iloc[0]

# Define potential layers to analyze for proximity
potential_layer_keys = [
    #'bepo:Gemarkungen_Abwasser',
    #'bepo:Gemarkungen_Abwaerme_Industrie',
    'bepo:Gemarkungen_Flussthermie',
    'bepo:Gemarkungen_Seethermie'
]

# Initialize an empty list to store close potential dataframes
all_close_potentials_list = []

for layer_key in potential_layer_keys:
    potential_type = layer_key.split(':')[1]  # Extract the type from the layer key
    close_potentials = find_close_potentials(layer_data[layer_key], selected_system_geom, max_distance)
    close_potentials = close_potentials.copy()  # To avoid SettingWithCopyWarning
    close_potentials['type'] = potential_type  # Add a column to label the potential type
    all_close_potentials_list.append(close_potentials)

# Concatenate all dataframes in the list into a single GeoDataFrame
all_close_potentials = pd.concat(all_close_potentials_list, ignore_index=True)

all_close_potentials = all_close_potentials[all_close_potentials['WaermePot'] > 0]

# Save the output to a file
#all_close_potentials.to_file(snakemake.output.gpkg, driver="GPKG")

# Assuming selected_system and all_close_potentials are already defined as per your script

# Plotting the district heating system
ax = selected_system.plot(color='red', figsize=(10, 10), label='District Heating System')

# Plotting the close potentials
all_close_potentials.plot(ax=ax, column='type', categorical=True, legend=True, alpha=0.7)

# Setting plot title and labels
plt.title('District Heating System and Close Potentials')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Showing the plot
plt.show()
