from owslib.wfs import WebFeatureService
import geopandas as gpd
import pandas as pd
import io
import matplotlib.pyplot as plt
import os
import yaml

with open('/Users/tomkaehler/Documents/Uni/BA/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

"""
params for snakemake
"""
# Snakemake parameter for selected system and max distance
selected_system_id = config['scenario']['selected_system_id']

# Access the max_distance value out of config file
max_distance = config['data_energieportal']['max_distance']


# Function to fetch data from the WFS service and load into a GeoDataFrame
def fetch_layer_and_save(wfs, layer_name, filepath, target_crs=None):
    if not os.path.exists(filepath):  # Check if file already exists to avoid re-downloading
        response = wfs.getfeature(typename=layer_name, outputFormat='json')
        data = gpd.read_file(io.BytesIO(response.read()))  # Directly load into GeoDataFrame

        if target_crs and data.crs != target_crs:  # Reproject only if necessary
            data = data.to_crs(target_crs)

        data.to_file(filepath, driver='GeoJSON')

url = 'https://energieportal-brandenburg.de/geoserver/bepo/ows'
wfs = WebFeatureService(url=url, version='2.0.0')

layer_info = {
    'bepo:Brandenburg_Fernwaerme': '/Users/tomkaehler/Documents/Uni/BA/data/energieportal/brandenburg_fernwaerme.json',
    'bepo:Gemarkungen_Flussthermie': '/Users/tomkaehler/Documents/Uni/BA/data/energieportal/gemarkungen_flussthermie.json',
    'bepo:Gemarkungen_Seethermie': '/Users/tomkaehler/Documents/Uni/BA/data/energieportal/gemarkungen_seethermie.json',
}

# Layers to be used
#layer_keys = [
#    #'bepo:Gemarkungen_Abwasser',
#    #'bepo:Gemarkungen_Abwaerme_Industrie',
#    #'bepo:Eignung_EWK',
#    #'bepo:Fläche_EWK',
#    'bepo:Brandenburg_Fernwaerme',
#    'bepo:Gemarkungen_Flussthermie',
#    'bepo:Gemarkungen_Seethermie',
#    #'bepo:datenquellen',
#    #'bepo:Ausschuss'
#]

for key, filepath in layer_info.items():
    fetch_layer_and_save(wfs, key, filepath, 'EPSG:4326')


def load_layer_from_geojson(filepath, filter_waermepot=False):
    gdf = gpd.read_file(filepath)
    # Optionally filter the GeoDataFrame based on 'WaermePot' if specified
    if filter_waermepot:
        gdf = gdf[gdf['WaermePot'] > 0]
        _ = gdf.sindex
    return gdf


# Adjust the dictionary comprehension to include conditional filtering
layer_data = {
    key: load_layer_from_geojson(
        filepath,
        filter_waermepot=('Gemarkungen' in key)  # Apply filtering based on layer key
    ) for key, filepath in layer_info.items()
}


def find_close_potentials(potential_gdf, system_geom, distance):
    # Convert the GeoDataFrames to EPSG:3035 for distance calculations
    potential_gdf_3035 = potential_gdf.to_crs("EPSG:3035")
    system_geom_3035 = system_geom.to_crs("EPSG:3035").geometry.iloc[0]
    # Perform the buffer operation in EPSG:3035
    search_area = system_geom_3035.buffer(distance)
    close_potentials = potential_gdf_3035[potential_gdf_3035.intersects(search_area)]
    # Optionally, convert back to EPSG:4326 if needed for plotting or further usage
    close_potentials = close_potentials.to_crs("EPSG:4326")
    return close_potentials


# Select the specific district heating system
fernwaerme_df = layer_data['bepo:Brandenburg_Fernwaerme']
selected_system = fernwaerme_df[fernwaerme_df['id'] == selected_system_id]
selected_system.to_file(f"/Users/tomkaehler/Documents/Uni/BA/output/selected_system_{selected_system_id}.gpkg", driver="GPKG")
#selected_system.to_file(snakemake.output.selected_system)

print(fernwaerme_df['geometry'].head())

# Define potential layers to analyze for proximity
potential_layer_keys = [
    'bepo:Gemarkungen_Flussthermie',
    'bepo:Gemarkungen_Seethermie'
]

# Initialize an empty list to store close potential dataframes
all_close_potentials_list = []

for layer_key in potential_layer_keys:
    potential_type = layer_key.split(':')[1]  # Extract the type from the layer key
    close_potentials = find_close_potentials(layer_data[layer_key], selected_system, max_distance)
    close_potentials = close_potentials.copy()  # To avoid SettingWithCopyWarning
    close_potentials['type'] = potential_type  # Add a column to label the potential type
    all_close_potentials_list.append(close_potentials)

# Concatenate all dataframes in the list into a single GeoDataFrame
all_close_potentials = pd.concat(all_close_potentials_list, ignore_index=True)
all_close_potentials.to_file(f"/Users/tomkaehler/Documents/Uni/BA/output/all_close_potentials_{selected_system_id}.gpkg", driver="GPKG")
#all_close_potentials.to_file(snakemake.output.all_close_potentials)

max_potentials_indices = all_close_potentials.groupby('type')['WaermePot'].idxmax()
max_potentials = all_close_potentials.loc[max_potentials_indices]
max_potentials.to_file(f"/Users/tomkaehler/Documents/Uni/BA/output/max_potentials_{selected_system_id}.gpkg", driver="GPKG")
#max_potentials.to_file(snakemake.output.max_potentials)


# Assuming selected_system and all_close_potentials are already defined as per your script
""""
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
"""
