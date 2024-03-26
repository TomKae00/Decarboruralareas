import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scripts.cost_functions import save_data_to_file
from scripts.cost_functions import load_data

from scripts.cost_functions import combined_data


import yaml

# Load the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def load_data_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Define variable params for config and snakemake
selected_system_id = config['scenario']['selected_system_id']
#selected_system_id = snakemake.params.system_id
year_of_interest = config['scenario']['year_of_interest']
#year_of_interest = snakemake.params.year_of_interest
supply_temp = config['scenario']['supply_temp'] + 273.15
#supply_temp = snakemake.params.supply_temp
return_temp = config['scenario']['return_temp'] + 273.15
#supply_temp = snakemake.params.return_temp
costs_data_year = config['scenario']['costs_data_year']
#costs_data_year = snakemake.params.costs_data_year

# fixed variables
project_year = config['cost_functions']['project_year']
cp = config['cost_functions']['cp']
rho = config['cost_functions']['rho']
dT = supply_temp - return_temp
component_parameters_config = config['component_parameters']


power_law_models_df = load_data(f'output/power_law_models_{selected_system_id}_{year_of_interest}_supply:{supply_temp}.csv')
erzeugerpreisindex = load_data(f'output/erzeugerpreisindex_{selected_system_id}_{year_of_interest}_supply:{supply_temp}.csv')


zuwachs_water_tank = erzeugerpreisindex.loc['Andere Behaelter f. fluessige Stoffe,aus Eisen,Stahl', project_year] / 100

# Extract component parameters from config and filter for specific storage types
storage_type = ['PTES', 'central water tank storage']  # List the storage types you're interested in
# Initialize component_parameters with entries for specific storage types only
component_parameters = {k: v for k, v in component_parameters_config.items() if k in storage_type}


# Iterate through each row in combined_data to apply parameters
for index, row in combined_data.iterrows():
    technology = row['Technology']
    fluid = row['fluid']

    # Use technology as the key for technologies without a specific fluid
    key = technology if fluid == 'NA' else f"{technology}_{fluid}"

    # Apply technology-specific parameters from the config, default to empty dict if not specified
    params = component_parameters_config.get(technology, {})

    component_parameters[key] = params


def generate_cost_function(alpha, beta):
    """
    Generates a power law cost function for a given component based on alpha and beta parameters.

    Parameters:
    - alpha: Coefficient of the power law model.
    - beta: Exponent of the power law model.

    Returns:
    - A cost function that takes a single argument x and returns the cost based on the power law model.
    """

    def cost_function(x):
        return np.where(x > 0, beta * x ** alpha, 0)

    return cost_function


def cost_per_kWh_PTES(x, dT, cp, rho):
    # Calculate cost only where x > 0, otherwise return 0
    return np.where(x > 0, (((0.9 + 2.44 * 10**(-5) * x) * 10**6) / (x * dT * cp * rho)) * 10**6, 0)


def cost_per_kWh_central_water_tank_storage(x, dT, cp, rho):
    # Calculate cost only where x > 0, otherwise return 0
    return np.where(x > 0, ((7450 * x ** (-0.47)) * zuwachs_water_tank * x) / (x * dT * cp * rho) * 10 ** 6, 0)


cost_calculation_map = {
    'PTES': cost_per_kWh_PTES,
    'central water tank storage': cost_per_kWh_central_water_tank_storage,
}


def make_storage_cost_function(dT, cp, rho, storage_type):
    def storage_cost_function(x):
        if storage_type in cost_calculation_map:
            # Retrieve the specific cost function from the map based on storage_type
            specific_cost_function = cost_calculation_map[storage_type]
            return specific_cost_function(x, dT, cp, rho)
    return storage_cost_function


cost_functions = {}

for storage_type in cost_calculation_map.keys():
    cost_functions[storage_type] = make_storage_cost_function(dT, cp, rho, storage_type)

for component, row in power_law_models_df.iterrows():  # component is obtained from the index directly
    alpha = row['alpha']
    beta = row['beta']
    if component not in cost_functions:  # Check to avoid overwriting PTES/hot water tank
        cost_functions[component] = generate_cost_function(alpha, beta)


def constant_cost_approx_with_dynamic_segments(cost_function, lower_limit, upper_limit, error_threshold):
    def calculate_average_error_for_intervals(num_intervals):
        log_intervals = np.linspace(np.log(lower_limit), np.log(upper_limit), num_intervals + 1)
        intervals = np.exp(log_intervals)

        constant_costs = []
        all_errors = []
        for i in range(len(intervals) - 1):
            start, end = intervals[i], intervals[i + 1]
            x_interval = np.linspace(start, end, 100)
            y_interval = cost_function(x_interval)
            average_cost = np.mean(y_interval)
            constant_costs.append(average_cost)
            errors = np.abs(y_interval - average_cost)
            all_errors.extend(errors)

        return np.mean(all_errors), intervals, constant_costs

    # Initial setup
    num_intervals = 5  # Start with a minimum number of intervals
    while True:
        average_error, intervals, constant_costs = calculate_average_error_for_intervals(num_intervals)
        if average_error <= error_threshold:
            break
        num_intervals += 1

    return num_intervals, average_error, intervals, constant_costs


# Now, let's integrate this function into the overall process for generating piecewise constant approximations
def apply_piecewise_approx_to_individual_dfs(cost_functions, component_parameters):
    all_dfs = {}  # Dictionary to store a DataFrame for each component

    for component, cost_function in cost_functions.items():
        lower_limit = component_parameters[component]["lower_limit"]
        upper_limit = component_parameters[component]["upper_limit"]
        error_threshold = component_parameters[component]["error_threshold"]

        num_intervals, average_error, intervals, constant_costs = constant_cost_approx_with_dynamic_segments(
            cost_function, lower_limit, upper_limit, error_threshold
        )

        # Prepare data for DataFrame: Start and End capacities with corresponding Constant Costs
        data = {
            'Start Capacity': intervals[:-1],  # Exclude the last value for 'start'
            'End Capacity': intervals[1:],  # Exclude the first value for 'end'
            'Constant Cost': constant_costs
        }

        # Create a DataFrame for the current component
        component_df = pd.DataFrame(data)

        # Store the DataFrame in the dictionary
        all_dfs[component] = component_df

        # Optional: Plotting for each component
        x_range = np.linspace(lower_limit, upper_limit, 1000)
        plt.figure(figsize=(12, 8))
        plt.plot(x_range, cost_function(x_range), label='Original Function', color='blue', linewidth=2)
        for i in range(num_intervals):
            start = intervals[i]
            end = intervals[i + 1] if i < num_intervals - 1 else upper_limit
            plt.hlines(constant_costs[i], start, end, colors='red', linestyles='dashed', linewidth=2,
                       label='Constant Cost Approximation' if i == 0 else "")
        plt.xlabel('Capacity (kW)')
        plt.ylabel('Specific Cost ($/kW)')
        plt.title(
            f'{component}: Cost Function vs. Constant Cost Approximation\n{num_intervals} Segments, Avg Error: {average_error:.2f}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.show()

    return all_dfs


# Generate the DataFrames
component_dfs = apply_piecewise_approx_to_individual_dfs(cost_functions, component_parameters)

all_technologies_dfs = {}

cop_series = load_data_from_file(f'output/cop_series_{selected_system_id}_{year_of_interest}_supply:{supply_temp}.pkl')

storage_type = ['PTES', 'central water tank storage']

for component_name, df in component_dfs.items():

    if component_name in storage_type:
        df['Start Capacity'] = df['Start Capacity'] * rho * dT * cp * 1e-6  # Convert to MWh
        df['End Capacity'] = df['End Capacity'] * rho * dT * cp * 1e-6
    else:
        df['Start Capacity'] = df['Start Capacity'] / 1000  # Convert kW to MW
        df['End Capacity'] = df['End Capacity'] / 1000
        df['Constant Cost'] = df['Constant Cost'] * 1000        # Convert kW to MW

    # Split component_name to identify if it has a specific fluid associated (for heat pumps)
    parts = component_name.split("_")
    technology = parts[0] if parts else None
    fluid = "_".join(parts[1:]) if len(parts) > 1 else None

    if technology in cop_series and fluid in cop_series[technology]:
        df['cop_series'] = f"{technology}_{fluid}"
    else:
        df['cop_series'] = 'NA'  # Use 'NA' for technologies without specific fluids

    # Prepare the DataFrame entry
    df['Technology'] = technology  # Add 'Technology' column for consistency

    if technology in all_technologies_dfs:
        all_technologies_dfs[technology].append(df)
    else:
        all_technologies_dfs[technology] = [df]

technologies_to_remove = []

for technology_name, dfs in all_technologies_dfs.items():
    if config['components_enabled'].get(technology_name, False):
        combined_df = pd.concat(dfs, ignore_index=True)
        max_end_capacity = combined_df['End Capacity'].max()
        combined_df['M'] = max_end_capacity
        combined_df.set_index('Technology', inplace=True)
        all_technologies_dfs[technology_name] = combined_df
    else:
        technologies_to_remove.append(technology_name)

# Now, remove the technologies that are not enabled
for technology_name in technologies_to_remove:
    all_technologies_dfs.pop(technology_name, None)

save_data_to_file(all_technologies_dfs, f'output/all_technologies_dfs_{selected_system_id}_{year_of_interest}_supply:{supply_temp}_return:{return_temp}.pkl')