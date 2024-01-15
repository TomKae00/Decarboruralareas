from owslib.wfs import WebFeatureService
import geopandas as gpd
from owslib.etree import etree
import io

# URL des WFS-Dienstes
# bepo alle daten aus simenergy
# energierzeugungsanlagen
# solaratlas

url = 'https://energieportal-brandenburg.de/geoserver/bepo/ows'

# Verbindung zum WFS herstellen
wfs = WebFeatureService(url=url, version='2.0.0')

# Verfügbare Layer auflisten (optional)
for layer in wfs.contents:
    print(layer, wfs[layer].title, wfs[layer].abstract)

layer = 'bepo:Brandenburg_Netze'
layer_info = wfs[layer]

print(layer_info.title, layer_info.abstract, layer_info.boundingBoxWGS84)

#filterxml = etree.tostring(filter_property.toXML()).decode("utf-8")

# Request data from the layer
response = wfs.getfeature(typename=layer, outputFormat='json')

# Read the data
data = response.read()

data_io = io.BytesIO(data)

Netze = gpd.read_file(data_io)
