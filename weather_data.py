import atlite
import logging

import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.animation as animation

import pandas as pd

# for getting information on what is going on
logging.basicConfig(level=logging.INFO)
# for debugging better use this one -> more information
# logging.basicConfig(level=logging.DEBUG)

cutout = atlite.Cutout(
    path="Germany-2019.nc",
    module="era5",
    x=slice(4.8479, 15.8638),
    y=slice(55.2353, 46.9644),
    time=slice("2019-01", "2019-12")
)

module = "era5"
slice("2019-01", "2019-12")
x = slice(4.8479, 15.8638) # Longitude
y = slice(55.2353, 46.9644) # Latitude
cutout.prepare()

# weiter gucken was dies kann übernommen von https://github.com/PyPSA/pypsa-eur/blob/f3832521c424ddebd33c50a35aecfc9979707bdf/scripts/build_cutout.py#L43-L79

#logging.getLogger(__name__)
#if __name__ == "__main__":
#    if "snakemake" not in globals():
#        from _helpers import mock_snakemake

# get data
temp = cutout.data.temperature
print(temp)

# Funktion für die Aktualisierung des Plots
def update(frame):
    plt.clf()
    temp.isel(time=frame).plot(x='x', y='y', cmap='viridis')
    plt.title(f'Temperatur bei {temp.time[frame].values}')

# Erstelle die Animation
ani = animation.FuncAnimation(plt.figure(), update, frames=len(temp.time), repeat=False)
plt.show()

x_point = 13.7043
y_point = 51.6359

# Wähle die Daten für den spezifischen Punkt aus
temp_at_point = temp.sel(x=x_point, y=y_point, method='nearest')

# Wandele die Xarray-Daten in eine Pandas Series um
temperature_series = temp_at_point.to_series() - 273.15

# Zeige die erstellte Serie an
print(temperature_series)

# change temperature_series from K to °C

temperature_series_celsius = temperature_series - 273.15

print(temperature_series_celsius)

wind = cutout.data.wnd_azimuth
wind_at_point = wind.sel(x=x_point, y=y_point, method='nearest')
wind_series = wind_at_point.to_series()
print(wind_series)
