import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cost_functions import power_law_models_df
from cost_functions import cop_series
from cost_functions import temperature_series_map

heat_pumps_params = {
    "lower_limit": 50,
    "upper_limit": 15000,
    "error_threshold": 30  # Default for demonstration, adjust as necessary
}

component_parameters = {
    "central electric boiler": {"lower_limit": 150, "upper_limit": 20000, "error_threshold": 5},
    "electrolysis AEC": {"lower_limit": 5000, "upper_limit": 1e6, "error_threshold": 5},
    "electrolysis PEMEC": {"lower_limit": 5000, "upper_limit": 1e6, "error_threshold": 5},
    "central gas boiler": {"lower_limit": 10, "upper_limit": 20000, "error_threshold": 5},
    "central H2 boiler": {"lower_limit": 10, "upper_limit": 20000, "error_threshold": 5},
    # Add other components with lower_limit as needed
}

for source in temperature_series_map.keys():
    for fluid in ["R290", "R1234yf", "R134a", "R600a", "R600", "R245fa", "R717"]:
        combined_key = f"{source}_{fluid}"
        component_parameters[combined_key] = heat_pumps_params.copy()


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


cost_functions = {}
for component, row in power_law_models_df.iterrows():  # component is obtained from the index directly
    alpha = row['alpha']
    beta = row['beta']
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

for component_name, df in component_dfs.items():
    parts = component_name.split("_")
    fluid = "_".join(parts[1:]) if len(parts) > 1 else None
    source = parts[0] if parts else None


    # Check for a matching COP series in a nested dictionary structure
    if source in cop_series and fluid in cop_series[source]:
        df['cop_series'] = f"{source}_{fluid}"  # Use a combined key for clarity

# Aggregate DataFrames based on source technology and rename
source_to_technology_name = {
    "Luft": "central air sourced heat pump",
    "Flussthermie": "central sourced-water heat pump",
    "Seethermie": "central sourced-sea heat pump",
}

all_technologies_dfs = {}

for component_name, df in component_dfs.items():

    df['Start Capacity'] = df['Start Capacity'] / 1000  # Convert kW to MW
    df['End Capacity'] = df['End Capacity'] / 1000  # Convert kW to MW
    df['Constant Cost'] = df['Constant Cost'] * 1000  # Convert €/kW to €/MW

    source = component_name.split("_")[0]

    if source in source_to_technology_name:
        technology_name = source_to_technology_name[source]
        df['Technology'] = technology_name  # Add 'Technology' column before aggregation
        df['fluid'] = "_".join(component_name.split("_")[1:])  # Ensure 'fluid' column is added
        if technology_name not in all_technologies_dfs:
            all_technologies_dfs[technology_name] = [df]
        else:
            all_technologies_dfs[technology_name].append(df)
    else:
        df['Technology'] = component_name  # Use the component name directly for non-source technologies
        all_technologies_dfs[component_name] = [df]

# Combine the DataFrames for each technology and set 'Technology' as index
for technology_name, dfs in all_technologies_dfs.items():
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.set_index('Technology', inplace=True)
    all_technologies_dfs[technology_name] = combined_df
