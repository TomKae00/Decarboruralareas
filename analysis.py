import pandas as pd
import matplotlib.pyplot as plt

from BA import heat_demand_series

# Adjust the paths according to your exported results
generator_results = pd.read_csv('results/generators.csv')
load_results = pd.read_csv('results/loads.csv')
storage_results = pd.read_csv('results/stores.csv')
links_results = pd.read_csv('results/links.csv')
# Add any other results files you need

# Example: Summing up capital and operational costs
# This is a simplified example - adjust according to your data structure
total_capital_costs = generator_results['capital_cost'].sum()
total_operational_costs = generator_results['marginal_cost'].sum()
# Add other financial calculations as needed

# Example: Plotting heat demand
heat_demand_series.plot()
plt.title('Heat Demand Over Time')
plt.xlabel('Time')
plt.ylabel('Heat Demand (Units)')
plt.show()

# Example: Plotting power generation from wind
wind_power_series.plot()
plt.title('Wind Power Generation Over Time')
plt.xlabel('Time')
plt.ylabel('Power Generation (Units)')
plt.show()

# Example: Summarizing heat production by heat pumps
heat_pump_production = storage_results[storage_results['name'].str.contains('heat_pump')]['p']
total_heat_production = heat_pump_production.sum()
# Add other analyses as needed

# Example: Exporting a summary DataFrame
summary_df = pd.DataFrame({
    'Total Capital Costs': [total_capital_costs],
    'Total Operational Costs': [total_operational_costs],
    # Include other summary metrics
})
summary_df.to_csv('analysis_summary.csv')


