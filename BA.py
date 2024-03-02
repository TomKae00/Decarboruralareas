# import packages
import pypsa
import pandas as pd
import yaml

from weather_data import temperature_series
from Wind_power import wind_series_for_pypsa
from linearizing_costs import all_technologies_dfs
from heat_demand import thh_series
from data_Energieportal3 import max_potentials
from cost_functions import cop_series

# so nicht gut gelöst. Gucken dass diese Werte in def eingebracht werden und dann in einem Skript für die Parameter festgelegt werden
T_out = temperature_series
VLT = 70
RLT = 35
dT_HP = 4  # Abkühlung der Quelltemperatur
dT = VLT - 40

start_date = "2019-01-01"
end_date = "2019-12-31 23:00"

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

params = config['params']

# Accessing individual parameters
fill_values = params['fill_values']
r = params['r']
nyears = params['nyears']
year = params['year']

cost_file = f"data/costs_{params['year']}.csv"


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def prepare_costs(cost_file, params, nyears):
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )

    costs = costs.fillna(params["fill_values"])

    def annuity_factor(v):
        return calculate_annuity(v["lifetime"], params["r"]) + v["FOM"] / 100

    costs["annuity_factor"] = costs.apply(annuity_factor, axis=1)

    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * nyears for i, v in costs.iterrows()
    ]  # .itterows macht genau das gleiche wie .apply

    return costs


result = prepare_costs(cost_file, params, params["nyears"])
print(result)
result.to_csv('data/result.csv', index=True)

for technology_name, component_df in all_technologies_dfs.items():
    if technology_name in result.index:
        if technology_name in result.index:
            tech_params = result.loc[technology_name]
            # Update parameters directly without iterating over rows if possible
            for param in ['efficiency', 'VOM', 'FOM', 'lifetime', 'efficiency-heat']:
                if param in tech_params:
                    component_df[param] = tech_params[param]


        def annuity_factor(v):
            return calculate_annuity(v["lifetime"], params["r"]) + v["FOM"] / 100


        # Calculate and update annuity for each component
        component_df['annuity_factor'] = component_df.apply(annuity_factor, axis=1)

        component_df["fixed"] = [
            annuity_factor(v) * v["Constant Cost"] * nyears for i, v in component_df.iterrows()
        ]  # .itterows macht genau das gleiche wie .apply

for technology_name, df in all_technologies_dfs.items():
    new_index = [f"{technology_name.replace(' ', '_').lower()}_{i + 1}" for i in range(len(df))]
    df.insert(0, 'name', new_index, True)
    df.set_index('name', inplace=True)
    all_technologies_dfs[technology_name] = df


def calculate_cop(technology_df, cop_series):
    cop_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="h"))
    for name, row in technology_df.iterrows():
        parts = row['cop_series'].split('_')
        if len(parts) >= 2:
            source_type, fluid = parts[0], '_'.join(parts[1:])
            if source_type in cop_series and fluid in cop_series[source_type]:
                series_list = cop_series[source_type][fluid]
                if series_list and series_list[0].index.equals(cop_df.index):
                    cop_df[name] = series_list[0]
    return cop_df


def update_capacity(technology_df, cop_df):
    for name, row in technology_df.iterrows():
        if name in cop_df.columns:
            min_cop = cop_df[cop_df[name] > 0][name].min()
            technology_df.at[name, 'Start Capacity'] /= min_cop
            technology_df.at[name, 'End Capacity'] /= min_cop


def calculate_average_waermepot(max_potentials, type_name, start_date, end_date, freq):
    """
    Create a Pandas Series for a specified type with timestamps as the index and constant values.

    Parameters:
    - max_potentials: DataFrame containing the rows with maximum 'WaermePot' for each type.
    - type_name: The type of potential for which to create the series ('Flussthermie' or 'Seethermie').
    - start_date: Start date for the date range.
    - end_date: End date for the date range.
    - freq: Frequency for the date range.

    Returns:
    A Pandas Series for the specified type.
    """

    if type_name in max_potentials['Art'].values:
        # Filter for the specified type
        max_potential_row = max_potentials[max_potentials['Art'] == type_name]

        # Assuming there's only one max value per type, get the 'WaermePot' value
        waermepot_value = max_potential_row['WaermePot'].iloc[0]

        # Create the date range
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)

        # Calculate the constant value for 'WaermePot' divided by total hours in the range
        total_hours = len(timestamps)
        average_waermepot = waermepot_value / total_hours

        # Create and return the series
        return average_waermepot
    else:
        print(f"Skipping '{type_name}' as it is not present in max_potentials.")  # Raise a comment
        return None  # Indicate absence of data for this type


# To get the series for 'Flussthermie'
flussthermie_value = calculate_average_waermepot(max_potentials, 'Flussthermie', start_date, end_date, "h")

# To get the series for 'Seethermie'
seethermie_value = calculate_average_waermepot(max_potentials, 'Seethermie', start_date, end_date, "h")

"""""
At this point the creation of the network starts. 
"""""

# create PyPSA network
n = pypsa.Network()
# set time steps
n.set_snapshots(pd.date_range(start=start_date, end=end_date, freq="h"))
sns = n.snapshots

# add a bus/region for electricity
n.add("Bus", "power")
# add a bus for district heating
n.add("Bus", "heat")

# add generator for wind generation
n.add("Generator",
      "windfarm",
      bus="power",
      capital_cost=result.loc["onwind", "investment"],
      marginal_cost=result.loc["onwind", "VOM"],
      p_nom_extendable=True,
      p_max_pu=wind_series_for_pypsa)

n.add("Bus", "hot water storage")

n.add('Link',
      'water tank charger',
      bus0='heat',
      bus1='hot water storage',
      efficiency=result.loc["water tank charger", "efficiency"],
      p_nom_extendable=True
      )

n.add('Link',
      'water tank discharger',
      bus0='hot water storage',
      bus1='heat',
      efficiency=result.loc["water tank discharger", "efficiency"],
      p_nom_extendable=True
      )

# add hot water Storage
n.add("Store",
      "water tank",
      bus="hot water storage",
      capital_cost=result.loc["central water tank storage", "fixed"],
      e_cyclic=True,
      e_nom_extendable=True,
      standing_loss=0.2 / 24)  # für die standing_loss recherche betreiben. was angenommen werden kann -> Formel suchen in der Literatur Recheche

n.add("Bus", "battery_storage")

n.add(
    "Link",
    "battery charger",
    bus0='power',
    bus1='battery_storage',
    carrier="battery charger",
    # the efficiencies are "round trip efficiencies"
    efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
    capital_cost=result.loc["battery inverter", "fixed"],
    p_nom_extendable=True
)

# add battery storage battery inverter
n.add("Store",
      "battery store",
      bus="battery_storage",
      e_cyclic=True,
      capital_cost=result.loc["battery storage", "fixed"],
      efficiency=result.loc["battery storage", "efficiency"],
      e_nom_extendable=True
      )

n.add(
    "Link",
    "battery discharger",
    bus0='battery_storage',
    bus1='power',
    carrier="battery discharger",
    efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
    p_nom_extendable=True
)

# add H2 storage
n.add("Bus", "H2")
n.add('Bus', 'Excess_heat')
# n.add("Bus", "H2_storage")


electrolysis_AEC = all_technologies_dfs["electrolysis AEC"]
n.madd("Link",
       electrolysis_AEC.index,
       bus0=["power"],
       bus1=["H2"],
       bus2=['Excess_heat'],
       capital_cost=electrolysis_AEC["fixed"],
       efficiency=electrolysis_AEC["efficiency"],
       efficiency2=electrolysis_AEC["efficiency-heat"],
       p_nom_extendable=True
       )

electrolysis_PEMEC = all_technologies_dfs["electrolysis PEMEC"]
n.madd("Link",
       electrolysis_PEMEC.index,
       bus0=["power"],
       bus1=["H2"],
       bus2=['Excess_heat'],
       capital_cost=electrolysis_PEMEC["fixed"],
       efficiency=electrolysis_PEMEC["efficiency"],
       efficiency2=electrolysis_PEMEC["efficiency-heat"],
       p_nom_extendable=True
       )

"""""
Die Annahme stimmt soweit bus0 beschreibt mein input und jeglicher weiterer Bus outputs bzw. input wenn negativ. 
Die efficiency wird in das Verhältnis gesetzt zum bus0 
"""""
eh_electrolysis_hp = all_technologies_dfs["central excess-heat heat pump"]
cop_eh_hp = calculate_cop(eh_electrolysis_hp, cop_series)
update_capacity(eh_electrolysis_hp, cop_eh_hp)

n.madd('Link',
       eh_electrolysis_hp.index,
       bus0=['Excess_heat'],
       bus1=['heat'],
       bus2=['power'],
       capital_cost=eh_electrolysis_hp["fixed"],
       efficiency=(1 / (cop_eh_hp - 1)) * cop_eh_hp,
       efficiency2=-1 / (cop_eh_hp - 1),
       p_nom_extendable=True
       )

river_hp = all_technologies_dfs["central sourced-water heat pump"]
cop_river_hp = calculate_cop(river_hp, cop_series)
update_capacity(river_hp, cop_river_hp)

n.add('Bus',
      'river_withdrawal'
      )

n.add('Generator',
      'river_potential',
      bus='river_withdrawal',
      p_nom_max=flussthermie_value,
      p_nom_extendable=True
      )

n.madd('Link',
       river_hp.index,
       bus0=['river_withdrawal'],
       bus1=['heat'],
       bus2=['power'],
       capital_cost=river_hp["fixed"],
       efficiency=(1 / (cop_river_hp - 1)) * cop_river_hp,
       efficiency2=-1 / (cop_river_hp - 1),
       p_nom_extendable=True
       )

if "central sourced-sea heat pump" in all_technologies_dfs:
    lake_hp = all_technologies_dfs["central sourced-sea heat pump"]
    cop_lake_hp = calculate_cop(lake_hp, cop_series)
    update_capacity(lake_hp, cop_lake_hp)

    n.add('Bus',
          'lake_withdrawal'
          )

    n.add('Generator',
          'lake_potential',
          bus='lake_withdrawal',
          p_nom_max=seethermie_value,
          p_nom_extendable=True
          )

    n.madd('Link',
           river_hp.index,
           bus0=['river_withdrawal'],
           bus1=['heat'],
           bus2=['power'],
           capital_cost=river_hp["fixed"],
           efficiency=(1 / (cop_lake_hp - 1)) * cop_lake_hp,
           efficiency2=-1 / (cop_lake_hp - 1),
           p_nom_extendable=True
           )

n.add("Store",
      "H2 Speicher",
      bus="H2",
      e_cyclic=True,
      capital_cost=result.loc["hydrogen storage tank type 1 including compressor", "fixed"],
      e_nom_extendable=True)

central_H2_boiler = all_technologies_dfs["central H2 boiler"]
n.madd("Link",
       central_H2_boiler.index,
       bus0=["H2"],
       bus1=["heat"],
       p_nom_extendable=True,
       carrier=["H2 turbine"],
       efficiency=central_H2_boiler["efficiency"],
       capital_cost=central_H2_boiler["fixed"],
       marginal_cost=central_H2_boiler["VOM"]
       )

central_air_hp = all_technologies_dfs["central air sourced heat pump"]
cop_air_hp = calculate_cop(central_air_hp, cop_series)
update_capacity(central_air_hp, cop_air_hp)

n.madd("Link",
       central_air_hp.index,
       bus0=["power"],  # Assuming all heat pumps are connected to the same bus
       bus1=["heat"],  # Assuming all heat pumps supply to the same bus
       p_nom_extendable=True,
       efficiency=cop_air_hp,
       capital_cost=central_air_hp["fixed"]
       )

central_electric_boiler = all_technologies_dfs["central electric boiler"]
# add e-boiler
n.madd("Link",
       central_electric_boiler.index,
       bus0=["power"],
       bus1=["heat"],
       efficiency=central_electric_boiler["efficiency"],
       capital_cost=central_electric_boiler["fixed"],
       marginal_cost=central_electric_boiler["VOM"],
       p_nom_extendable=True)


def create_all_components_df(all_technologies_dfs):
    # Use a list comprehension to gather all DataFrames from the dictionary
    df_list = [df for df in all_technologies_dfs.values()]

    # Concatenate all the DataFrames in the list into one DataFrame
    all_components_df = pd.concat(df_list)

    return all_components_df


# Now, call the function with your dictionary
all_components_df = create_all_components_df(all_technologies_dfs)


def custom_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[all_components_df.index], name="Link-build", binary=True)
    # Retrieve the binary variable
    binary_var = n.model["Link-build"]
    print(binary_var)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[all_components_df.index]
    print(p_nom_opt)

    p_nom_min = all_components_df["Start Capacity"].loc[all_components_df.index]
    print(p_nom_min)

    m = all_components_df["M"].loc[all_components_df.index]
    print(m)

    for comp in all_components_df.index:
        # Apply the constraints
        n.model.add_constraints(p_nom_opt[comp] - binary_var[comp] * m[comp] <= 0,
                                name=f"Link-p_nom-upperlimit-{comp}")
        n.model.add_constraints(p_nom_opt[comp] - binary_var[comp] * m[comp] >= p_nom_min[comp] - m[comp],
                                name=f"Link-p_nom-lowerlimit-{comp}")


# add demand
n.add("Load",
      "heat_demand",
      bus="heat",
      p_set=thh_series)

# Specify the solver options
solver_options = {
    'gurobi': {
        'BarHomogeneous': 1  # Setting BarHomogeneous parameter for Gurobi
    }
}

c = 'link'  # Replace with the actual component name

if 'build_opt' not in n.links.columns:
    n.links['build_opt'] = 0  # Initialize with a default value (e.g., 0)

marginal_price_bus = n.buses_t.marginal_price

n.optimize(n.snapshots, extra_functionality=custom_constraints,
           solver_name="gurobi", solver_options=solver_options)
print("Optimization completed successfully")

# binary_results = n.model["Link-build"].solution.to_pandas()


# optimal capacities wind warm
print(n.generators.p_nom_opt)
# dispatch wind farm
print(n.generators_t.p)
n.generators_t.p.plot()
# optimal capacities district heating store
print(n.stores.e_nom_opt)
# energy in store
n.stores_t.e.plot()

# plot results (mal gucken was das kann und wofür mann das braucht)
# n.generators_t.p.plot()
# n.plot()

# get statistics (mal gucken was das kann und wofür mann das braucht)
statistics = n.statistics().dropna()
curtailment = n.statistics.curtailment()  # nicht klar was dies macht

n.statistics.energy_balance()

n.export_to_csv_folder("results")
print("csv exported sucessfully")

links = n.links
components = n.components

thh_sum = thh_series.sum()

total_system_cost = n.objective
print("Total System Cost:", total_system_cost)
print(n.investment_period_weightings)
waermegestehungskosten = total_system_cost / thh_sum
print("Wärmegestehungskosten:", waermegestehungskosten)

# Netzverlsute intgrieren
