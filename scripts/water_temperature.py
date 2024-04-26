import pandas as pd
import yaml
import snakemake


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# variable params for config and snakemake
year_of_interest = config['scenario']['year_of_interest']
#year_of_interest = snakemake.params.year_of_interest


# Load the CSV file
file_path = 'data/HV_0120_Temperatur.csv'
water_temperature = pd.read_csv(file_path)

# Convert 'Datum' column to datetime format
water_temperature['Datum'] = pd.to_datetime(water_temperature['Datum'], dayfirst=True, format='%d.%m.%y')

# Sort the water_temperature by date to ensure chronological order
water_temperature.sort_values('Datum', inplace=True)

# Set the main period of interest through variables
start_date = f'{year_of_interest}-01-01'
end_date = f'{year_of_interest}-12-31 23:00'

# Calculate the extended range to include surrounding months for accurate edge interpolation
# Here, we extend the start date by one month back and the end date by one month forward
extended_start = pd.to_datetime(start_date) - pd.DateOffset(months=1)
extended_end = pd.to_datetime(end_date) + pd.DateOffset(months=1)

# Create the extended hourly range based on the above calculations
extended_range = pd.date_range(start=extended_start, end=extended_end, freq='h')

# Prepare the original water_temperature for interpolation by setting the date as index
water_temperature.set_index('Datum', inplace=True)

# Reindex the water_temperature to include all hours in the extended range, then interpolate
water_temperature_extended = water_temperature.reindex(extended_range)
water_temperature_interpolated = water_temperature_extended.interpolate(method='time')

# Focus on the specified period for the final output
water_temperature = water_temperature_interpolated.loc[start_date:end_date]
water_temperature = water_temperature['Temperatur (Wasser; OW-G)']
water_temperature_original = water_temperature
water_temperature_original.to_csv(f'output/water_temperature_original_{year_of_interest}.csv', index=True)

water_temperature_adjusted = water_temperature_original
water_temperature_adjusted.loc['2018-02-20 09:00:00':'2018-03-06 09:00:00'] = 2.9
water_temperature_adjusted.to_csv(f'output/water_temperature_adjusted_{year_of_interest}.csv', index=True)
