from owslib.wfs import WebFeatureService
import geopandas as gpd
import io
import pandas as pd


def fetch_layer(wfs, layer_name):
    response = wfs.getfeature(typename=layer_name, outputFormat='json')
    data = io.BytesIO(response.read())
    return gpd.read_file(data)


# URL of the WFS service
url = 'https://energieportal-brandenburg.de/geoserver/bepo/ows'

# Connect to the WFS
wfs = WebFeatureService(url=url, version='2.0.0')

for layer in wfs.contents:
    print(layer, wfs[layer].title, wfs[layer].abstract)

# Layers to be used
layers = [
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
layer_data = {layer: fetch_layer(wfs, layer) for layer in layers}

# Example: Access and work with a specific layer
fernwaerme_df = layer_data['bepo:Brandenburg_Fernwaerme']
abwaerme_df = layer_data['bepo:Gemarkungen_Abwaerme_Industrie']
abwasser_df = layer_data['bepo:Gemarkungen_Abwasser']
flussthermie_df = layer_data['bepo:Gemarkungen_Flussthermie']
ausschluss_df = layer_data['bepo:Ausschuss']
datenquellen_df = layer_data['bepo:datenquellen']
# Perform data analysis or visualization with netze_df