import os
import re
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"]#,
#    "font.size": 26  # Match the font size used in your LaTeX document
})

fontsize = 16
# Function to process the existing scenarios with subdirectories
def process_files_with_subdir(base_directory, subdirectory):
    directory = os.path.join(base_directory, subdirectory).rstrip('/')
    pattern = re.compile(r"supply:(\d+\.\d+).*?elecprice_(\d+).*?(redispatch)?\.csv$")
    results_redispatch = []
    results_no_redispatch = []

    files = os.listdir(directory)
    print(f"Processing {subdirectory} in {base_directory}. Total files found: {len(files)}")

    for filename in files:
        if not filename.endswith('.csv'):
            continue
        match = pattern.search(filename)
        if match:
            supply_temp = float(match.group(1))
            elec_price = int(match.group(2))
            redispatch = bool(match.group(3))
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)

            if 'waermegestehungskosten' not in data.columns:
                continue

            warm_cost = data['waermegestehungskosten'].iloc[0]
            record = {'SupplyTemp': supply_temp, 'ElecPrice': elec_price, 'WarmCost': warm_cost}

            if redispatch:
                results_redispatch.append(record)
            else:
                results_no_redispatch.append(record)

    df_redispatch = pd.DataFrame(results_redispatch).pivot_table(index='SupplyTemp', columns='ElecPrice', values='WarmCost', aggfunc='mean').iloc[::-1]
    df_no_redispatch = pd.DataFrame(results_no_redispatch).pivot_table(index='SupplyTemp', columns='ElecPrice', values='WarmCost', aggfunc='mean').iloc[::-1]

    return {'no_redispatch': df_no_redispatch, 'redispatch': df_redispatch}

# Function to process the additional scenario without subdirectories
def process_additional_scenario(directory):
    pattern = re.compile(r".*gasprice_(\d+)_CO2price(\d+).*\.csv")
    results = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            match = pattern.search(filename)
            if match:
                gas_price = int(match.group(1))
                co2_price = int(match.group(2))
                file_path = os.path.join(directory, filename)
                data = pd.read_csv(file_path)
                warm_cost = data['waermegestehungskosten'].iloc[0]  # Assuming we need the first row only

                results.append({'GasPrice': gas_price, 'CO2Price': co2_price, 'WarmCost': warm_cost})

    # Convert list of dicts into DataFrame
    df = pd.DataFrame(results)
    df = df.groupby(['GasPrice', 'CO2Price']).agg({'WarmCost': 'mean'}).reset_index()
    df_pivot = df.pivot(index='CO2Price', columns='GasPrice', values='WarmCost')
    df_pivot = df_pivot.iloc[::-1]

    return df_pivot

# Define main directories and subdirectories
main_directories = [dir.strip('/') for dir in ['Data/E-Kessel/', 'Data/Elektrolyse/', 'Data/WP/']]
sub_directories = ['HWS', 'Batterie']

# Collect data from all directories
data_summary = {}
for main_dir in main_directories:
    clean_dir = os.path.basename(main_dir)
    data_summary[clean_dir] = {}
    for sub_dir in sub_directories:
        data_summary[clean_dir][sub_dir] = process_files_with_subdir(main_dir, sub_dir)

# Add the additional scenario data
additional_scenario_directory = 'Data/Referenzvariante/'
additional_scenario_data = process_additional_scenario(additional_scenario_directory)
data_summary['Referenzvariante'] = {'NoSubDir': {'no_redispatch': additional_scenario_data}}

# Plotting function
def plot_data(data_summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'HWS': 'blue', 'Batterie': 'red', 'NoSubDir': 'grey'}
    markers = {'no_redispatch': 'o', 'redispatch': '*'}
    markersize = 35
    labels_added = set()

    print("Starting to plot data...")

    # Calculate the x positions for each main directory
    x_positions = {main_dir: idx for idx, main_dir in enumerate(data_summary.keys())}

    for main_dir, subdirs in data_summary.items():
        for sub_dir, data_types in subdirs.items():
            x_base = x_positions[main_dir]
            for data_type, data in data_types.items():
                for col in data.columns:
                    for idx in data.index:
                        y_value = data.loc[idx, col]
                        offset = 0.1 if sub_dir == "Batterie" else -0.1
                        if sub_dir == "NoSubDir": offset = 0.0
                        x_position = x_base + offset
                        label = f"{sub_dir} - {data_type}"
                        if label not in labels_added:
                            labels_added.add(label)
                        else:
                            label = ""  # Set label to empty string for already used labels
                        ax.scatter(x_position, y_value, color=colors[sub_dir], marker=markers.get(data_type, 'x'), label=label, s=markersize)

    ax.set_xticks([x for x in x_positions.values()])
    ax.set_xticklabels([os.path.basename(dir) for dir in x_positions.keys()], rotation=0)
    ax.set_ylabel('WGK[€/MWh]', fontsize=fontsize)
#    ax.set_title('Comparison of WarmCost across Scenarios')
#    ax.legend(title="Scenarios", loc='upper right')
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.set_ylim(0, 300)
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color='grey'#, zorder=10
            )
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    plt.savefig('all_prices.eps', format='eps', dpi=300)
    plt.show()



# Plot the data
plot_data(data_summary)
