import pandas as pd
import geopandas as gpd
from fuzzywuzzy import process
import re
import yaml
import snakemake

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# variable params for config and snakemake
selected_system_id = config['scenario']['selected_system_id']
#selected_system_id = snakemake.params.system_id
year_of_interest = config['scenario']['year_of_interest']
#year_of_interest = snakemake.params.year_of_interest


# fixed settings
base_temperature = config['heat_demand']['base_temperature']  # Base temperature for HDD calculation
hot_water_demand_percentage = config['heat_demand']['hot_water_demand_percentage']  # Percentage of total heat demand attributed to hot water
heating_season_start = config['heat_demand']['heating_season_start']  # September
heating_season_end = config['heat_demand']['heating_season_end']  # April

# gleich nochmal überprüfen
network_loss_percentage = 15


# fixed Configuration Variables
json_file_path = 'data/energieportal/brandenburg_fernwaerme.json'
csv_file_path = 'data/energieportal/waermebedarf_fernwaerme_2022.csv'

# Load the district heating data out of the fixed configuration variable and convert to pd.DataFrame
brandenburg_fernwaerme_data = gpd.read_file(json_file_path)
brandenburg_fernwaerme_df = pd.DataFrame(brandenburg_fernwaerme_data)

# Load the heat demand data of the csv file
waermebedarf_fernwaerme_2022_df = pd.read_csv(csv_file_path, sep=';')


# Function to get the best match
def get_best_match(row, choices):
    betreiber = row['Betreiber']
    best_match = process.extractOne(betreiber, choices)
    return best_match[0]  # Return only the best match string

# Function to lookup heat demand
def lookup_heat_demand(best_match_gebiet, lookup_df):
    pattern = re.escape(best_match_gebiet)
    matched_row = lookup_df[lookup_df['Gebiet'].str.contains(pattern, na=False, case=False, regex=True)]
    if not matched_row.empty:
        return matched_row['Heizungstechnik: Wärmenetz Wärmebedarf (kWh)'].values[0]
    else:
        return pd.NA


# Applying the matching function
gebiet_choices = waermebedarf_fernwaerme_2022_df['Gebiet'].unique()

brandenburg_fernwaerme_df['Best_Match_Gebiet'] = brandenburg_fernwaerme_df.apply(get_best_match, choices=gebiet_choices, axis=1)
brandenburg_fernwaerme_df['heat_demand_kWh'] = brandenburg_fernwaerme_df['Best_Match_Gebiet'].apply(lambda x: lookup_heat_demand(x, waermebedarf_fernwaerme_2022_df))

# Filter for the selected district heating system
selected_system = brandenburg_fernwaerme_df[brandenburg_fernwaerme_df['id'] == selected_system_id]

temperature_series2022 = pd.read_csv(f'output/temperature_series2022_{selected_system_id}.csv', index_col=0)
temperature_series2022 = temperature_series2022['temperature']
temperature_series2022.index = pd.to_datetime(temperature_series2022.index)

temp_df_2022 = pd.DataFrame(temperature_series2022, columns=['temperature'])

# Calculate HDD
temp_df_2022['HDD'] = base_temperature - temp_df_2022['temperature']
temp_df_2022['HDD'] = temp_df_2022['HDD'].apply(lambda x: max(x, 0))

temperature_series = pd.read_csv(f'output/temperature_series_{selected_system_id}_{year_of_interest}.csv', index_col=0)
temperature_series = temperature_series['temperature']
temperature_series.index = pd.to_datetime(temperature_series.index)

temp_df = pd.DataFrame(temperature_series, columns=['temperature'])

# Calculate HDD
temp_df['HDD'] = base_temperature - temp_df['temperature']
temp_df['HDD'] = temp_df['HDD'].apply(lambda x: max(x, 0))

total_hdd_2022 = temp_df_2022['HDD'].sum()
total_hdd = temp_df['HDD'].sum()

ratio_hdd = total_hdd / total_hdd_2022

THD_2022 = selected_system['heat_demand_kWh'].iloc[0]
SH_Y = THD_2022 * ratio_hdd * (1 - hot_water_demand_percentage/100)

# Determine if each date is within the heating season
temp_df['is_heating_season'] = ((temp_df.index.month >= heating_season_start) |
                                (temp_df.index.month <= heating_season_end))

# Assuming 'heat_demand_kWh' is correctly assigned to 'selected_system' from earlier steps

# Calculate hourly hot water demand (constant throughout the year)
hours_in_year = 8760
HW_H = ((SH_Y / (1 - hot_water_demand_percentage / 100)) * hot_water_demand_percentage / 100) / hours_in_year
#HL_H = (((SH_Y + HW_H.sum()) / (1 - network_loss_percentage / 100)) * network_loss_percentage / 100) / hours_in_year

# Initialize the THH column for total hourly heat demand
temp_df['THH'] = 0

# Calculate THH using vectorized operations
total_hdd_heating_season = temp_df[temp_df['is_heating_season']]['HDD'].sum()
temp_df['SHH'] = temp_df.apply(lambda x: (x['HDD'] / total_hdd_heating_season * SH_Y) if x['is_heating_season'] else 0, axis=1)
temp_df['HWH'] = HW_H

temp_df['THH'] = (temp_df['SHH'] + HW_H) / ((100 - network_loss_percentage)/100)

temp_df['HLH'] = temp_df['THH'] - temp_df['SHH'] - HW_H

thh_series = temp_df['THH'] / 1000 # convert to MWh
print(thh_series.sum())

thh_series.to_csv(f'output/thh_series_{selected_system_id}_{year_of_interest}.csv', index=True)

