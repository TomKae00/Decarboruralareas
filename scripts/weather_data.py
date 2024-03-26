import atlite
import logging
import geopandas as gpd
import snakemake

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


logging.basicConfig(level=logging.INFO)

for year in range(2014, 2023):
    cutout = atlite.Cutout(
        path=f"input/cutout_germany/germany_{year}.nc",
        module="era5",
        x=slice(4.8479, 15.8638),
        y=slice(55.2353, 46.9644),
        time=slice(f"{year}-01", f"{year}-12"),
    )

    cutout.prepare(features=["wind", "temperature"])

    logging.info(f"Completed cutout for {year}")

"""
defining the params with or without snakemake
"""
# this is the way to read the config via snakemake (maybe even without ['scenario']['year_of_interest'] need to figure this out)
#year_of_interest = snakemake.config['scenario']['year_of_interest']
selected_system_id = config['scenario']['selected_system_id']
year_of_interest = config['scenario']['year_of_interest']
#selected_system = gpd.read_file(snakemake.input.selectd_system)

"""
preparation of the temperature series for the selescted system id based on the cutout year chosen 
"""

path = f"input/cutout_germany/germany_{year_of_interest}.nc"
#path = snakemake.input.cutout

selected_system = gpd.read_file(f"output/selected_system_{selected_system_id}.gpkg")


geometry = selected_system.geometry.iloc[0]

centroid = geometry.centroid
x_point, y_point = centroid.x, centroid.y

# Load the cutout for the specified year
cutout = atlite.Cutout(path=path)


temp = cutout.data.temperature

temp_at_point = temp.sel(x=x_point, y=y_point, method='nearest')


temperature_series = temp_at_point.to_series() - 273.15
temperature_series.to_csv(f"output/temperature_series_{selected_system_id}_{year_of_interest}.csv", index=True)
#temperature_series.to_csv(snakemake.output.temp)

#load the temperature series for the year 2022 to get the ratio of HDD in relation to the selected temperature series above
path_2022 = "input/cutout_germany/germany_2022.nc"

cutout_2022 = atlite.Cutout(path=path_2022)

temp_2022 = cutout_2022.data.temperature

temp_at_point_2022 = temp_2022.sel(x=x_point, y=y_point, method='nearest')

temperature_series2022 = temp_at_point_2022.to_series() - 273.15
temperature_series2022.to_csv(f"output/temperature_series2022_{selected_system_id}.csv", index=True)
#temperature_series2022.to_csv(snakemake.output.temp2022)

