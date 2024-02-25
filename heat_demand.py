import pandas as pd
import geopandas as gpd
from fuzzywuzzy import process
import re
# Assuming weather_data is a module you've defined that contains the temperature_series
from weather_data import temperature_series

selected_district_heating_id = 'Brandenburg_Fernwaerme.3'  # The ID of the district heating system to analyze
base_temperature = 15  # Base temperature for HDD calculation
hot_water_demand_percentage = 25  # Percentage of total heat demand attributed to hot water
heating_season_start = 9  # September
heating_season_end = 4  # April
network_loss_percentage = 15


# Configuration Variables
json_file_path = 'data/energieportal/brandenburg_fernwaerme.json'
csv_file_path = 'data/energieportal/waermebedarf_fernwaerme_2022.csv'

# Load the district heating data
brandenburg_fernwaerme_data = gpd.read_file(json_file_path)
brandenburg_fernwaerme_df = pd.DataFrame(brandenburg_fernwaerme_data)

# Load the heat demand data
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
selected_system = brandenburg_fernwaerme_df[brandenburg_fernwaerme_df['id'] == selected_district_heating_id]

pd.to_datetime(temperature_series.index)

temp_df = pd.DataFrame(temperature_series, columns=['temperature'])

# Calculate HDD
temp_df['HDD'] = base_temperature - temp_df['temperature']
temp_df['HDD'] = temp_df['HDD'].apply(lambda x: max(x, 0))

# Determine if each date is within the heating season
temp_df['is_heating_season'] = ((temp_df.index.month >= heating_season_start) |
                                (temp_df.index.month <= heating_season_end))

# Assuming 'heat_demand_kWh' is correctly assigned to 'selected_system' from earlier steps
SH_Y = selected_system['heat_demand_kWh'].iloc[0]

# Calculate hourly hot water demand (constant throughout the year)
hours_in_year = 8760
HL_H = ((SH_Y / (1 - network_loss_percentage / 100)) * network_loss_percentage / 100) / hours_in_year
HW_H = ((SH_Y / (1 - hot_water_demand_percentage / 100)) * hot_water_demand_percentage / 100) / hours_in_year

# Initialize the THH column for total hourly heat demand
temp_df['THH'] = 0

# Calculate THH using vectorized operations
total_hdd_heating_season = temp_df[temp_df['is_heating_season']]['HDD'].sum()
temp_df['SHH'] = temp_df.apply(lambda x: (x['HDD'] / total_hdd_heating_season * SH_Y) if x['is_heating_season'] else 0, axis=1)
temp_df['HLH'] = HL_H
temp_df['HWH'] = HW_H

temp_df['THH'] = temp_df['SHH'] + temp_df['HLH'] + temp_df['HWH']

thh_series = temp_df['THH'] / 1000 # convert to MWh