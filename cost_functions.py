import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from weather_data import temperature_series
from data_Energieportal3 import all_close_potentials


"""
Implementieren what the flip flop meine Bivalenztemperatur oder Auslegungstemperatur ist. Diese kann dabei minimale
Außentemperatur sein, eine beliebige temperatur oder die minimale vorgeschriebene Auslegungstemperatur. Darauf basierend 
werden die Kosten festgelegt, da die Wärmepumpen Leistung für diesen Punkt festgelegt ist. Dann COP an diesem Punkt
bestimmen und mit diesem durch die MW_th teilen. NICHT durch die Kosten teilen, da der output MW_th witerhin ist. Also
doch gucken welche Vorschriften es nach der EN 14511 gibt und ab wann man von der Leistung MW_th bei Wärmepumpen
spricht (also bei welchen Bedingungen). Doch bullshit, ich muss mit T_supply festlegen bei wlechen Bedingungen 1000kW 
erreiche. Die perfekte Auslegung hieraus zu erkennen als Ausblick definieren!
"""


def load_data(file_path):
    """Load data from CSV file and set the first column as index."""
    return pd.read_csv(file_path, index_col=0)


# Load the data
parameters_tci = load_data('data/table_3_data.csv')
# parameters_cop = load_data('data/table_4_data.csv')
costs_data = pd.read_csv('data/costs_function_2025.csv')

unique_sources_from_df = all_close_potentials['Art'].unique()
available_sources = np.unique(np.append(unique_sources_from_df, ["Luft", "Abwaerme"]))

temperature_series_outside = temperature_series  # Pandas series for outside temperature
temperature_series_flussthermie = temperature_series + 5  # Pandas series for Flussthermie temperature
temperature_series_seethermie = temperature_series + 7  # Pandas series for Seethermie temperature
temperature_series_abwaerme = pd.Series(50, index=temperature_series.index)

# Map of source types to their temperature series
temperature_series_map = {
    'Luft': temperature_series_outside,
    'Flussthermie': temperature_series_flussthermie,
    'Seethermie': temperature_series_seethermie,
    'Abwaerme': temperature_series_abwaerme
}

min_temp_thresholds = {
    'Luft': 0,  # Example threshold
    'Flussthermie': 0,
    'Seethermie': 0,
    'Abwaerme': 0  # Adjust these values as needed
}


T_supply_k = 60 + 273.15

# Auslegung der Quelltemperautr minimal für die Kosten der Wärmepumpen
# Ausschalttemperautr für Luft implementieren 5°C, 0°, -5°, -10° als varibale integrieren Designtemperatur und gucken wie Ergebnisse sich verändern

def calculate_for_source(source_type):
    # Access the correct temperature series for the current source type
    T_source_c = temperature_series_map[source_type]
    T_source_c_min = T_source_c.min()

    min_temp_threshold = min_temp_thresholds[source_type]  # Get the threshold for the current source type
    if T_source_c_min < min_temp_threshold:
        T_source_c_min = min_temp_threshold

    dT_lift = T_supply_k - (T_source_c + 273.15)
    dT_lift_k_min = T_supply_k - (T_source_c_min + 273.15)
    COP_carnot = T_supply_k / dT_lift

    return T_source_c, T_source_c_min, dT_lift, dT_lift_k_min, COP_carnot


for source_type in available_sources:
    T_source_c, T_source_c_min, dT_lift, dT_lift_k_min, COP_carnot = calculate_for_source(source_type)


def calculate_intermediate_max_supply_temp_R717(T_source_c_min):
    if T_source_c_min < 51.20:
        T_supply_COP_max = 44.6 + 0.9928 * T_source_c_min
    else:
        T_supply_COP_max = 104.20 - 0.1593 * T_source_c_min
    if T_source_c_min < 60.35:
        T_supply_TCI_min = 28.50 + 0.859 * T_source_c_min
    else:
        T_supply_TCI_min = 87.26 - 0.1102 * T_source_c_min
    T_supply_intermediate = (T_supply_COP_max + T_supply_TCI_min) / 2
    return T_supply_intermediate


# Functions for Calculations
"""""
R717 needs to be calculated for T_source_c so for every component given. Needs to be adjusted at the moment not correct
"""""
def calculate_max_supply_temperatures(T_source_c_min):
    """Calculate and return the max supply temperatures for various fluids including R717."""
    temps = {'R290': 99.7, 'R134a': 101.7, 'R600a': 137.3, 'R600': 154.9, 'R245fa': 154.8, #'R1234yf': 95.3,
             'R717': calculate_intermediate_max_supply_temp_R717(T_source_c_min)}
    # Convert to Kelvin
    return {fluid: temp + 273.15 for fluid, temp in temps.items()}


max_supply_temps_k = calculate_max_supply_temperatures(T_source_c_min)


def check_constraints(fluid, T_supply_k):
    if fluid == 'R717':
        T_supply_intermediate = calculate_intermediate_max_supply_temp_R717(T_source_c_min)
        return T_supply_k <= max_supply_temps_k[fluid] and T_supply_k <= T_supply_intermediate
    else:
        return fluid in max_supply_temps_k and T_supply_k <= max_supply_temps_k[fluid]


def calculate_tci_strich_1000(D, E, F, T_supply_k, dT_lift_k_min):
    """Calculate TCI for a 1000 kW heat pump."""
    return D + E * dT_lift_k_min + F * T_supply_k


# Calculation Functions
def calculate_scaled_tci(TCI_strich_1000, alpha, X_kW_th):
    """Scale TCI based on heat pump size."""
    return TCI_strich_1000 * (X_kW_th / 1000) ** alpha


def calculate_COP_1000(G, H, I, J, K, T_supply_k, dT_lift, T_source_c, min_temp_threshold):
    """Calculate COP based on parameters, supply/dT_lift conditions, and a minimum temperature threshold."""
    COP_series = pd.Series(index=T_source_c.index, dtype=float)

    # Vectorized condition: Set COP to 0 if T_source_c is below the min_temp_threshold
    condition = T_source_c < min_temp_threshold
    COP_series[condition] = 0

    # For temperatures above the threshold, calculate the COP
    not_condition = ~condition
    COP_series[not_condition] = G + H * dT_lift[not_condition] + I * T_supply_k + J * dT_lift[not_condition]**2 + K * T_supply_k * dT_lift[not_condition]

    return COP_series

def calculate_additional_costs(size_mw):
    """Calculate construction, electricity, and heat source investment costs in euros."""
    construction_cost = (0.084311 * size_mw + 0.021769) * 1e6  # Convert from Mio. EUR to EUR
    electricity_cost = (0.12908 * size_mw + 0.01085) * 1e6  # Convert from Mio. EUR to EUR
    if source_type == 'Luft':
        heat_source_cost = (0.12738 * size_mw + 5.5007e-6) * 1e6
    elif source_type == 'Abwaerme':
        heat_source_cost = (0.091068 * size_mw + 0.10846) * 1e6
    elif source_type == 'Seethermie':
        heat_source_cost = (0.12738 * size_mw + 5.5007e-6) * 1e6 #keine richtigen WERTE sondern die für gw
    elif source_type == 'Flussthermie':
        heat_source_cost = (0.12738 * size_mw + 5.5007e-6) * 1e6 #keine richtigen WERTE sondern die für gw
    else:
        raise ValueError("Invalid source type")
    return construction_cost, electricity_cost, heat_source_cost


cop_series = {}
results = {}

# Main Calculation
sizes_kw_th = np.arange(0, 5001, 100)  # Heat pump sizes from 0 to 5000 kW_th

for source_type in available_sources:

    min_temp_threshold = min_temp_thresholds[source_type]
    # Calculate source-specific parameters
    (T_source_c, T_source_c_min, dT_lift, dT_lift_k_min, COP_carnot) = calculate_for_source(source_type)

    # Initialize storage for source-specific calculations
    results[source_type] = {}
    cop_series[source_type] = {}

    for fluid in parameters_tci.columns:
        if not check_constraints(fluid, T_supply_k):
            print(f"{fluid} cannot be used due to constraints.")
            continue
        D, E, F, G, H, I, J, K, alpha = parameters_tci.loc[['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'α'], fluid].values

        cop_series[source_type][fluid] = {'COP': []}  # Initialize COP list for the fluid
        COP_1000 = calculate_COP_1000(G, H, I, J, K, T_supply_k, dT_lift, T_source_c, min_temp_threshold)
        cop_series[source_type][fluid]['COP'].append(COP_1000)

        results[source_type][fluid] = {'TCI': [], 'Total Investment Costs': [], 'Specific Costs': []}
        for size_kw_th in sizes_kw_th:
            size_mw = size_kw_th / 1000  # Convert kW_th to MW

            TCI_strich_1000 = calculate_tci_strich_1000(D, E, F, T_supply_k, dT_lift_k_min)
            scaled_tci = calculate_scaled_tci(TCI_strich_1000, alpha, size_kw_th)

            construction_cost, electricity_cost, heat_source_cost = calculate_additional_costs(size_mw)
            total_investment_cost = scaled_tci + construction_cost + electricity_cost + heat_source_cost

            specific_cost = total_investment_cost / size_kw_th if size_kw_th != 0 else 0

            results[source_type][fluid]['TCI'].append(scaled_tci)
            results[source_type][fluid]['Total Investment Costs'].append(total_investment_cost)
            results[source_type][fluid]['Specific Costs'].append(specific_cost)

"""""
pre-screening analysis of the different fluids for the used sources by using the stroed data about the costs 
get data in dataframes and then print them later in analysis 
"""""
average_cop_source_fluid = {}
for source, fluids in cop_series.items():
    average_cop_source_fluid[source] = {}
    for fluid, data in fluids.items():
        # Calculate the average COP for this source and fluid
        average_cop_source_fluid[source][fluid] = np.mean(data['COP'])

mean_specific_costs_source_fluid = {}
for source_type, fluids in results.items():
    mean_specific_costs_source_fluid[source_type] = {}
    for fluid, data in fluids.items():
        # Calculate the mean specific cost for this source and fluid
        mean_specific_costs_source_fluid[source_type][fluid] = np.mean(data['Specific Costs'])

efficiency_cost_ratio_source_fluid = {}
for source, fluids in average_cop_source_fluid.items():
    efficiency_cost_ratio_source_fluid[source] = {}
    for fluid, avg_cop in fluids.items():
        mean_cost = mean_specific_costs_source_fluid[source][fluid]
        # Calculate efficiency-cost ratio for this source and fluid
        mean_ratio = avg_cop / mean_cost if mean_cost else 0
        efficiency_cost_ratio_source_fluid[source][fluid] = mean_ratio

# Step 3: Convert the efficiency-cost ratio to a DataFrame
#df_mean_ratio = pd.DataFrame.from_dict(efficiency_cost_ratio_mean) orient='index', columns=['Efficiency-Cost Ratio'])

#best_ratio = max(efficiency_cost_ratio_mean.values())

# Identify fluids within 10% of the best ratio

selected_fluids_per_source = {}
for source, fluids in efficiency_cost_ratio_source_fluid.items():
    best_ratio = max(fluids.values())
    tolerance = 0.05  # 10%
    selected_fluids_per_source[source] = {fluid: ratio for fluid, ratio in fluids.items() if ratio >= (1-tolerance) * best_ratio}

# Convert the selected fluids to a DataFrame for easier analysis and display
#df_selected_fluids = pd.DataFrame.from_dict(selected_fluids, orient='index', columns=['Efficiency-Cost Ratio within 10% Tolerance'])

#print(df_selected_fluids)

"""""
Mit new_technologies_list wird alles ready gemacht für den export in einen DataFrame, 
darunter die Aufteilung in die Kategorien von meinem dataframe data, der auf der csv costs_function_2025.csv beruht
Die Daten werden dann vorbereitet, um für die power_law_models genutzt zu werden. Weiterer 'for' loop dann für die 
Nutzung von den unterschiedlichen Kältemitteln für die gewählte source, die muss noch als Technologie umgenannt werden. 
So, dass diese dann in costs_2020.csv gefunden werden kann. 
"""""

new_technologies_list = []
for source_type, selected_fluids in selected_fluids_per_source.items():
    # Loop through each selected fluid for the source
    for fluid in selected_fluids.keys():
        # Access the specific cost data for the selected fluid and source_type
        if fluid in results[source_type]:  # Ensure the fluid is present in the results for the source
            fluid_data = results[source_type][fluid]
            for size, spec_cost in zip(sizes_kw_th, fluid_data['Specific Costs']):
                if size != 0:  # Filter out zero sizes if necessary
                    new_technologies_list.append({
                        'Technology': source_type,
                        'fluid': fluid,  # Note: 'Technology' is now considered as 'fluid'
                        'Size (kW)': size,
                        'Specific cost (EUR/kW)': spec_cost
                    })

# If you wish to convert this list to a DataFrame for further analysis or export
new_technologies_df = pd.DataFrame(new_technologies_list)

costs_data['Size (kW)'] = costs_data['Size (MW)'] * 1000
costs_data.drop('Size (MW)', axis=1, inplace=True)
# Append the new data to the existing DataFrame
combined_data = pd.concat([costs_data, new_technologies_df], ignore_index=True)
combined_data.to_csv('data/forGPT.csv')
#combined_data.set_index('Technology', inplace=True)


# Fit a power-law model for each technology
if 'fluid' not in combined_data.columns:
    combined_data['fluid'] = 'NA'  # Assign a default value for entries without a fluid

combined_data['fluid'].fillna('NA', inplace=True)

# Now, adapt the loop to handle both cases
power_law_models = {}
for (technology, fluid), group in combined_data.groupby(['Technology', 'fluid']):
    # Debug: Print technology and fluid being processed
    log_size = np.log(group['Size (kW)'])
    log_specific_cost = np.log(group['Specific cost (EUR/kW)'])
    X_log = log_size.values.reshape(-1, 1)
    y_log = log_specific_cost.values
    model = LinearRegression()
    model.fit(X_log, y_log)
    alpha = model.coef_[0]
    beta = np.exp(model.intercept_)

    # Handle the key naming based on whether a fluid is associated
    combined_key = f"{technology}" if fluid == 'NA' else f"{technology}_{fluid}"
    power_law_models[combined_key] = {'alpha': alpha, 'beta': beta}

# Convert power law models to a DataFrame for easier analysis or export
power_law_models_df = pd.DataFrame.from_dict(power_law_models, orient='index')
power_law_models_df.to_csv('data/power_law_models.csv')

"""""

# Visualize the cost function for a specific technology, e.g., Electric Boiler
tech = 'central electric boiler'
alpha = power_law_models[tech]['alpha']
beta = power_law_models[tech]['beta']
electric_boiler_data = data[data['Technology'] == tech]
x_range = np.linspace(min(electric_boiler_data['Size (kW)']), max(electric_boiler_data['Size (kW)']), 100)
y_range = beta * x_range ** alpha
plt.figure(figsize=(10, 6))
plt.scatter(electric_boiler_data['Size (kW)'], electric_boiler_data['Specific cost (EUR/kW)'], color='blue', label='Data Points')
plt.plot(x_range, y_range, color='red', label='Cost Function (EUR/kW)')
plt.xlabel('Size (kW)')
plt.ylabel('Specific Cost (EUR/kW)')
plt.title(f'{tech} Cost Function with Data Points (EUR/kW)')
plt.legend()
plt.grid(True)
plt.show()

"""""
