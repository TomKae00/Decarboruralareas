import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches


# Initialize an empty DataFrame to collect monthly negative price counts for each year
monthly_negative_counts = pd.DataFrame()

# Directory containing the data files
data_directory = "electricity_price/"
years_of_interest = range(2014, 2024)

# Process each year
for year in years_of_interest:
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31 23:00'
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    # Construct file path
    file_path = os.path.join(data_directory,
                             f"energy-charts_Stromproduktion_und_Boersenstrompreise_in_Deutschland_{year}.csv")

    # Read the CSV file
    electricity_market = pd.read_csv(file_path)
    electricity_market.index = timestamps

    # Drop the 'Datum (MEZ)' column if it exists
    if 'Datum (MEZ)' in electricity_market.columns:
        electricity_market = electricity_market.drop('Datum (MEZ)', axis=1)

    # Focus on the 'Day Ahead Auction Price' column
    electricity_market = electricity_market['Day Ahead Auktion Preis (EUR/MWh; EUR/tCO2)']

    # Calculate the occurrences of negative prices per month
    negative_prices = (electricity_market <= 0).resample('M').sum()
    negative_prices.name = year  # Rename series to the current year
    monthly_negative_counts = pd.concat([monthly_negative_counts, negative_prices], axis=1)

monthly_negative_counts.reset_index(inplace=True)

# Convert the index column (now a regular column named 'index') to datetime to extract the month
monthly_negative_counts[''] = pd.to_datetime(monthly_negative_counts['index']).dt.month

monthly_negative_counts.drop(columns='index', inplace=True)

# Instead of setting the index to '', create a pivot table
monthly_pivot = monthly_negative_counts.pivot_table(index='',
                                                    values=[x for x in range(2014, 2024)],
                                                    aggfunc='sum')

# Reset index to turn the '' index into a column
monthly_pivot.reset_index(inplace=True)

monthly_long = pd.melt(monthly_pivot, id_vars=[''], var_name='year', value_name='value')

# Map the '' numeric values to month abbreviations
month_map = {1: 'Jan', 2: 'Feb', 3: 'Mär', 4: 'Apr', 5: 'Mai', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dez'}
monthly_long['Monate'] = monthly_long[''].map(month_map)

# Drop the extra '' column now that we have 'Monate
monthly_long.drop('', axis=1, inplace=True)

# Ensure 'year' column is of type int if necessary
monthly_long['year'] = monthly_long['year'].astype(int)

# Rearrange columns to match df.csv
monthly_transformed = monthly_long[['year', 'Monate', 'value']]


# Now 'monthly_negative_counts' holds the count of negative price days per month for each year
# Plotting
sns.set(style="whitegrid")
#palette = sns.color_palette("mako", n_colors=12)  # 'mako' palette for a blue gradient effect

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 18
})

plt.figure(figsize=(9, 5))
ax = sns.boxplot(
    x='Monate',
    y='value',
    data=monthly_transformed,
    color='lightblue',
    order=["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"],
    whis=[5, 95],
    zorder=2  # Lower z-order so that swarmplot points appear on top
)

# Overlaying a swarmplot with higher z-order to appear above the boxplot
swarm = sns.swarmplot(
    x='Monate',
    y='value',
    data=monthly_transformed,
    color='0.25',  # Dark points for the swarmplot
    size=3,
    ax=ax,
    zorder=1  # Higher z-order ensures swarmplot is on top
)

# Adding titles and labels with custom font sizes
#ax.set_title('ly Negative Price Counts Over Years 2014-2023')
#ax.set_xlabel('Monat')
ax.set_ylabel('Anzahl der Stunden mit negativem Strompreis')

# Rotating the x-axis labels for better visibility
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)


#ax.spines['left'].set_position('zero')
ax.set_ylim(bottom=0)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_bounds(0, ax.get_ybound()[1])

# Only show ticks on the left spine for positive y-axis values
ax.yaxis.set_ticks_position('left')
ax.tick_params(axis='y', which='both', length=0)  # Hide y-axis ticks

# Draw a line for the x-axis at y=0
plt.axhline(0, color='black', linewidth=0.8)

plt.savefig('redispatch_2014_2023.eps', format='eps', dpi=1200)


plt.show()

# Outputting negative values per year data
negative_values_per_year = monthly_negative_counts.sum()
print(negative_values_per_year)

