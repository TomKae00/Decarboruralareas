import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import yaml

with open('/Users/tomkaehler/Documents/Uni/BA/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

selected_system_id = config['scenario']['selected_system_id']


# Initialize a dictionary to hold combined data for each series type
combined_data_dict = {
    'temperature': None,
    'thh': None  # This will hold the combined data for the THH series
}

# Define the series types you have
series_types = ['temperature', 'thh']

# Corresponding actual column names in the CSV files
column_names = {
    'temperature': 'temperature',  # Assuming this is correct for the temperature files
    'thh': 'THH'  # Corrected to the actual column name in the THH files
}

# Loop through the years
for year in range(2014, 2023):
    for series_type in series_types:
        # Construct the file path based on the series type
        file_path = f'output/{series_type}_series_{selected_system_id}_{year}.csv'

        # Read the CSV file
        df = pd.read_csv(file_path, parse_dates=['time'])

        # Convert date to month-day-hour format (without year)
        df['time'] = df['time'].dt.strftime('%m-%d-%H')

        # Check for leap years and remove February 29th
        if year in [2016, 2020]:
            # Remove all entries from February 29th for leap years
            df = df[~df['time'].str.startswith('02-29')]

        # Set this new time as the index
        df.set_index('time', inplace=True)

        # Use the correct column name for each series type
        actual_column_name = column_names[series_type]

        # Rename the relevant column to include both the series type and the year for easier identification
        df = df[[actual_column_name]].rename(columns={actual_column_name: f'{series_type}_{year}'})

        # Combine data from different years for the current series type
        if combined_data_dict[series_type] is None:
            combined_data_dict[series_type] = df
        else:
            combined_data_dict[series_type] = combined_data_dict[series_type].combine_first(df)

thh_sums = combined_data_dict['thh'].sum()

# Calculate the minimum temperature for each column
temperature_mins = combined_data_dict['temperature'].min()

average_thh_sum = thh_sums.mean()

# Calculate the average of the minimum temperatures
average_temperature_min = temperature_mins.mean()

# Display the results
print("THH Sum for Each Year:")
print(thh_sums)
print("\nMinimum Temperature for Each Year:")
print(temperature_mins)

# Display the averages
print("\nAverage THH Sum across Years:")
print(average_thh_sum)
print("\nAverage Minimum Temperature across Years:")
print(average_temperature_min)


def read_geopackage(filepath, target_crs=None):
    """
    Reads a GeoPackage file into a GeoDataFrame and optionally transforms its CRS.

    Parameters:
    - filepath (str): The path to the GeoPackage file.
    - target_crs (str, optional): The target CRS to transform the GeoDataFrame to.
      If None, the GeoDataFrame's current CRS is retained.

    Returns:
    - GeoDataFrame: The GeoDataFrame loaded from the GeoPackage file.
    """
    gdf = gpd.read_file(filepath)

    if target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


selected_system_id = config['scenario']['selected_system_id']

path_all_close_potentials = f"output/all_close_potentials_{selected_system_id}.gpkg"
path_max_potentials = f"output/max_potentials_{selected_system_id}.gpkg"
path_selected_system = f"output/selected_system_{selected_system_id}.gpkg"

# Read the GeoPackage files using the new function, converting to EPSG:3857 for plotting
all_close_potentials = read_geopackage(path_all_close_potentials, target_crs="EPSG:3857")
max_potentials = read_geopackage(path_max_potentials, target_crs="EPSG:3857")
selected_system = read_geopackage(path_selected_system, target_crs="EPSG:3857")

all_close_potentials = all_close_potentials.simplify(tolerance=0.1, preserve_topology=True)
max_potentials = max_potentials.simplify(tolerance=0.1, preserve_topology=True)

#all_close_potentials = all_close_potentials.geometry.centroid
#max_potentials = max_potentials.geometry.centroid

fig, ax = plt.subplots(figsize=(10, 10))

# Add the layers to the plot
#all_close_potentials.plot(ax=ax, color='blue', alpha=0.3, label='All Close Potentials')  # Adjusted alpha for visibility
#max_potentials.plot(ax=ax, color='red', marker='*', markersize=50, label='Max Potential')  # Adjusted markersize for distinction
selected_system.plot(ax=ax, color='red', marker='^', markersize=10, label='Selected System')  # Ensure this is plotted distinctly

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set the bounds
bounds = selected_system.total_bounds
margin = 0.2 * (bounds[2] - bounds[0])
ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
ax.set_ylim(bounds[1] - margin, bounds[3] + margin)

# Remove axis
ax.axis('off')

# Add legend
ax.legend()

plt.show()
