from owslib.wfs import WebFeatureService
import geopandas as gpd
import io
import snakemake


# Function to fetch data from the WFS service and load into a GeoDataFrame
def fetch_layer(wfs, layer_name):
    response = wfs.getfeature(typename=layer_name, outputFormat='json')
    data = io.BytesIO(response.read())
    return gpd.read_file(data) # alles wie zuvor


# Function to reproject GeoDataFrame to a given CRS
def reproject_to_crs(gdf, crs):
    return gdf.to_crs(crs) # noch angucken


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
    'bepo:Gemarkungen_Abwasser',
    'bepo:Gemarkungen_Abwaerme_Industrie',
    'bepo:Eignung_EWK',
    'bepo:Fläche_EWK',
    'bepo:Brandenburg_Fernwaerme',
    'bepo:Gemarkungen_Flussthermie',
    'bepo:Gemarkungen_Seethermie',
    'bepo:datenquellen',
    'bepo:Ausschuss'
]

# Fetch and store layers in a dictionary
layer_data = {key: fetch_layer(wfs, key) for key in layer_keys}

# Reproject layers to the specified CRS (EPSG:3035)
for key in layer_data:
    layer_data[key] = reproject_to_crs(layer_data[key], 'EPSG:3035')

# Snakemake parameters for selected system and max distance
selected_system_id = 'Brandenburg_Fernwaerme.7'  # snakemake.config['selected_system_id']
max_distance = 6000    # snakemake.config['max_distance']

# Select the specific district heating system
fernwaerme_df = layer_data['bepo:Brandenburg_Fernwaerme']
selected_system = fernwaerme_df[fernwaerme_df['id'] == selected_system_id] # richtiges Netz wird ausgewählt
selected_system_geom = selected_system.geometry.iloc[0]

# Define potential layers to analyze for proximity
potential_layer_keys = [
    'bepo:Gemarkungen_Abwasser',
    'bepo:Gemarkungen_Abwaerme_Industrie',
    'bepo:Gemarkungen_Flussthermie',
    'bepo:Ausschuss'
]

for key in layer_data:
    layer_data[key] = layer_data[key][layer_data[key].is_valid]

# Find and combine all close potentials into a single GeoDataFrame
all_close_potentials = gpd.GeoDataFrame(columns=['type', 'geometry'])
for layer_key in potential_layer_keys:
    potential_type = layer_key.split(':')[1]
    close_potentials = find_close_potentials(layer_data[layer_key], selected_system_geom, max_distance)
    print(f"Layer: {potential_type}, Close Potentials Found: {len(close_potentials)}")
    close_potentials.loc[:, 'type'] = potential_type
    all_close_potentials = all_close_potentials.append(close_potentials, ignore_index=True)

for layer_key in potential_layer_keys:
    print(f"Layer: {layer_key}, Number of Features: {len(layer_data[layer_key])}, Extent: {layer_data[layer_key].total_bounds}")

for layer_key in potential_layer_keys:
    invalid_geoms = layer_data[layer_key].geometry.isna() | ~layer_data[layer_key].is_valid
    print(f"Layer: {layer_key}, Invalid Geometries: {invalid_geoms.sum()}")

#all_close_potentials.to_file(snakemake.output.gpkg, driver="GPKG")

# Now `all_close_potentials` is a GeoDataFrame containing all types of potentials within the max distance
