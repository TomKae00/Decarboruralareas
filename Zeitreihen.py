# All currently utilised functions concerning data download, reading and writing

import atlite
import xarray as xr
import rasterio as rio
import rioxarray as rioxr

import os.path

import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
import geopandas as gpd

import cartopy.crs as ccrs
from cartopy.crs import PlateCarree as plate
import cartopy.io.shapereader as shpreader
import shapely
from shapely.geometry import shape, box

import requests

from vt2geojson.tools import vt_bytes_to_geojson



def agora_tile_to_dict(wind_dist, z, x, y):
    """
    Function that scrapes the agroa-windflächenrechner tileserver for one vectortile.

    Prameters
    ----------
    wind_dist = str, can be '400', '600', '800', '1000'
        Distance to settlement setting
    x,y,z = str
        Parameters of vectortile where:
        x = horizontal extent, y = vertical extent, z = zoom-level

    Returns
    -------
    dict
        Dictinary entry with x and y parameter plus the GeoJSON information on the vectortile-
    """

    url = "https://wfr.agora-energiewende.de/potential_area_mvt/{}/{}/{}/?setup__wind_distance={}".format(z, x, y,
                                                                                                          wind_dist)

    print(f'Downloading tile {x} / {y}')
    r = requests.get(url)

    try:
        assert r.status_code == 200, r.content

    except AssertionError:
        print(f'    Tile {x} / {y} not available')

    vt_content = r.content

    # Transformation of binary vectortiles into text-based GeoJson
    features = vt_bytes_to_geojson(vt_content, x, y, z)

    return {"x": x, "y": y, 'features': features}


agora = agora_tile_to_dict(wind_dist=1000, z=10, x=10, y=2)