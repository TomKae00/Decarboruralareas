import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"]#,
#    "font.size": 26  # Match the font size used in your LaTeX document
})


def process_all_scenarios(base_directory):
    pattern = re.compile(r"supply:(\d+\.\d+)_return.*elecprice_(\d+).*")
    results = {}
    for scenario_dir in os.listdir(base_directory):
        directory = os.path.join(base_directory, scenario_dir)
        if os.path.isdir(directory):
            match = pattern.search(directory.replace('/', ':'))
            if match:
                supply = float(match.group(1))
                elec_price = int(match.group(2))
                redispatch = 'redispatch' in directory
                scenario_key = f"supply_{supply}_elecprice_{elec_price}_{'redispatch' if redispatch else 'no_redispatch'}"
                results[scenario_key] = process_scenario(directory)
    return results

def process_scenario(directory):
    links_file = os.path.join(directory, 'links.csv')
    links_p1_file = os.path.join(directory, 'links-p1.csv')
    stores_file = os.path.join(directory, 'stores.csv')
    stores_p_file = os.path.join(directory, 'stores-p.csv')
    try:
        links = pd.read_csv(links_file)
        links_p1 = pd.read_csv(links_p1_file)
        stores = pd.read_csv(stores_file)
        stores_p = pd.read_csv(stores_p_file)
    except FileNotFoundError:
        print(f"Missing files in {directory}.")
        return None
    valid_links = links[(links['build_opt'] == 1) & (~links['name'].str.contains("electrolysis"))]['name']
    valid_stores = stores[stores['e_nom_opt'] > 0.1]['name']
    data_series_dict = {}
    for link_name in valid_links:
        if link_name in links_p1.columns:
            data_series_dict[link_name] = links_p1[link_name]
    for store_name in valid_stores:
        if store_name in stores_p.columns:
            data_series_dict[f"store_{store_name}"] = stores_p[store_name]
    return data_series_dict


base_directory = 'Data/E-Kessel/HWS'
data_by_scenario = process_all_scenarios(base_directory)

thh_path = 'Data/thh_series_Brandenburg_Fernwaerme.29_2018.csv'
thh_df = pd.read_csv(thh_path)
thh_series = thh_df['THH']

# Extract values with error checking
electricity_prices = []
supply_temperatures = []
for key in data_by_scenario:
    try:
        parts = key.split('_')
        supply_temp = float(parts[1].replace('supply', ''))
        elec_price = float(parts[3].replace('elecprice', ''))
        supply_temperatures.append(supply_temp)
        electricity_prices.append(elec_price)
    except ValueError as e:
        print(f"Error processing key {key}: {e}")

electricity_prices = sorted(set(electricity_prices))
supply_temperatures = sorted(set(supply_temperatures))
redispatch_statuses = [False, True]

# Create figure and axes for the subplots
structured_data = {
    (elec_price, supply_temp, redispatch): data_by_scenario.get(
        f"supply_{supply_temp}_elecprice_{elec_price}_{'redispatch' if redispatch else 'no_redispatch'}", {})
    for elec_price in electricity_prices
    for supply_temp in supply_temperatures
    for redispatch in redispatch_statuses
}

# Now we can proceed to plot using this structured data
rows = len(electricity_prices) * 2  # Two rows per electricity price (no redispatch and redispatch)
cols = len(supply_temperatures)


def plot_data(thh_series, data_by_scenario, scenario_key, ax):
    german_months = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                     "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

    # Assuming the series starts from January and is continuous through the year
    month_starts = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016]

    month_mids = [(month_starts[i] + month_starts[i + 1]) // 2 for i in range(len(month_starts) - 1)]
    month_mids.append((month_starts[-1] + 8760) // 2)

    inverted_thh_series = -thh_series.abs()
    pos_base = pd.Series(0, index=thh_series.index)
    ax.fill_between(thh_series.index, 0, inverted_thh_series, label='THH', color='tomato', alpha=0.8, linewidth=0)
    link_colors = {'central_electric_boiler_12': 'blue', 'central_electric_boiler_9': 'blue', 'central_air_sourced_heat_pump_11': 'skyblue',  'central_air_sourced_heat_pump_12': 'skyblue','central_air_sourced_heat_pump_14': 'skyblue', 'central_h2_boiler_1': 'seagreen',
                   'central_excess-heat_heat_pump_2': 'yellow', 'central_excess-heat_heat_pump_1': 'yellow', 'central_excess-heat_heat_pump_20': 'yellow'}
    if scenario_key in data_by_scenario:
        scenario_data = data_by_scenario[scenario_key]
        for series_name, series_data in scenario_data.items():
            if series_name.startswith('store_'):
                positive_values = series_data[series_data > 0]
                ax.fill_between(series_data.index, pos_base, pos_base + positive_values, label='WWS', color='darkorange', alpha=0.8, linewidth=0, )
                pos_base += positive_values
                negative_values = series_data[series_data < 0]
                ax.fill_between(series_data.index, inverted_thh_series, inverted_thh_series + negative_values, color='darkorange', alpha=0.8, linewidth=0)
            else:
                color = link_colors.get(series_name, 'steelblue')
                abs_series_data = series_data.abs()
                ax.fill_between(series_data.index, pos_base, pos_base + abs_series_data, label=series_name, alpha=0.8, color=color, edgecolor=color, linewidth=0)
                pos_base += abs_series_data

    ax.set_xticks(month_mids)  # Set custom x-ticks to the middle of each month
    ax.set_xticklabels(german_months, rotation=45, fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
#    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))

    #ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    #ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: german_months[mdates.num2date(x).month - 1]))
    #plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)  # Rotate labels

#    ax.set_xlabel('Time (Monthly data)')
#    ax.set_ylabel('Value', fontsize=fontsize)
#    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color='grey'#, zorder=10
            )
    ax.set_xlim(thh_series.index.min(), thh_series.index.max())
    ax.set_ylim(y_min, y_max)
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

fontsize = 20
y_min = -22
y_max = 22


# Assuming data_by_scenario and thh_series are defined, and plotting framework setup
fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(18, 30), sharex='col', sharey='row')

#temperature_mapping = {
#    343.15: r"\underline{VLT/RLT [°C]}\n{\underline{70/40}}",
#    363.15: r"\underline{VLT/RLT [°C]}\n{\underline{90/50}}",
#    383.15: r"\underline{VLT/RLT [°C]}\n{\underline{110/60}}"
#}

temperature_mapping = {
    343.15: r"\underline{VLT/RLT [°C]: 70/40}",
    363.15: r"\underline{VLT/RLT [°C]: 90/50}",
    383.15: r"\underline{VLT/RLT [°C]: 110/60}"
}

supply_temperatures_sorted = sorted(set(supply_temperatures))
for j, temp in enumerate(supply_temperatures_sorted):
#    axs[0, j].set_title(f'{temp}°C', fontsize=16)  # Set column titles on the first row
    new_label = temperature_mapping.get(temp, "Unknown")  # Default to "Unknown" if temp is not in the mapping
    axs[0, j].set_title(new_label, fontsize=fontsize, loc='center', pad=20)

electricity_prices_sorted = sorted(set(electricity_prices))
redispatch_statuses = [False, True]

for i, elec_price in enumerate(sorted(set(electricity_prices))):
    for j, supply_temp in enumerate(sorted(set(supply_temperatures))):
        for k, redispatch in enumerate([False, True]):
            row_index = 2 * i + k
            col_index = j
            ax = axs[row_index, col_index]
            redispatch_str = 'redispatch' if redispatch else 'no_redispatch'
            scenario_key = f"supply_{supply_temp}_elecprice_{int(elec_price)}_{redispatch_str}"
            print(scenario_key)
            plot_data(thh_series, data_by_scenario, scenario_key, ax)

            if col_index == 0:  # Only add labels to the first column to avoid repetition
                extra_text = "+ übersch. Strom" if redispatch else ""
                label = f'{int(elec_price)} €/MWh {extra_text}'
                ax.set_ylabel(r'\underline{' + label + '}', fontsize=fontsize, rotation=90, labelpad=40)

                ax2 = ax.twinx()
                ax2.set_yticks([])  # Remove ticks on secondary axis
                ax2.spines['right'].set_visible(False)  # Hide the spine on the right
                # Secondary label
                ax2.set_ylabel('Wärmeleistung [MW]', fontsize=fontsize, rotation=90, labelpad=42)  # Adjust labelpad as necessary
                ax2.yaxis.set_label_position('left')

                for spine in ax2.spines.values():
                    spine.set_linewidth(0.2)


plt.tight_layout()

plt.savefig('E-Kessel_HWS_all_time_series.eps', format='eps', dpi=300)
plt.show()

