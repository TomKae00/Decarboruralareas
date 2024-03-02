import pandas as pd
import matplotlib as plt

# Load the CSV file
file_path = 'data/HV_0120_Temperatur.csv'

data = pd.read_csv(file_path)

# Convert 'Datum' column to datetime format
data['Datum'] = pd.to_datetime(data['Datum'], dayfirst=True, format='%d.%m.%y')

# Sort the data by date to ensure chronological order
data.sort_values('Datum', inplace=True)

# Set the main period of interest through variables
start_date = "2019-01-01"
end_date = "2019-12-31 23:00"

# Calculate the extended range to include surrounding months for accurate edge interpolation
# Here, we extend the start date by one month back and the end date by one month forward
extended_start = pd.to_datetime(start_date) - pd.DateOffset(months=1)
extended_end = pd.to_datetime(end_date) + pd.DateOffset(months=1)

# Create the extended hourly range based on the above calculations
extended_range = pd.date_range(start=extended_start, end=extended_end, freq='h')

# Prepare the original data for interpolation by setting the date as index
data.set_index('Datum', inplace=True)

# Reindex the data to include all hours in the extended range, then interpolate
data_extended = data.reindex(extended_range)
data_interpolated = data_extended.interpolate(method='time')

# Focus on the specified period for the final output
data_final = data_interpolated.loc[start_date:end_date]
