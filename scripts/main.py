# import packages
import pypsa
import pandas as pd
import yaml
import geopandas as gpd
import pickle
import logging
import re

with open('/Users/tomkaehler/Documents/Uni/BA/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def load_data_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Define variable params for config and snakemake
selected_system_id = config['scenario']['selected_system_id']
# selected_system_id = snakemake.params.system_id
year_of_interest = config['scenario']['year_of_interest']
# year_of_interest = snakemake.params.year_of_interest
supply_temp = config['scenario']['supply_temp'] + 273.15
# supply_temp = snakemake.params.supply_temp
return_temp = config['scenario']['return_temp'] + 273.15
# return_temp = snakemake.params.return_temp
costs_data_year = config['scenario']['costs_data_year']
# costs_data_year = snakemake.params.costs_data_year
electricity_price = config['scenario']['electricity_price']
# electricity_price = snakemake.params.electricity_price
gas_price = config['scenario']['gas_price']
# gas_price = snakemake.params.gas_price
co2_price = config['scenario']['co2_price']
# co2_price = snakemake.params.co2_price
run = config['run']
project_year = config['cost_functions']['project_year']

max_potentials = gpd.read_file(f'/Users/tomkaehler/Documents/Uni/BA/output/max_potentials_{selected_system_id}.gpkg')
temperature_series_outside = pd.read_csv(f'/Users/tomkaehler/Documents/Uni/BA/output/temperature_series_{selected_system_id}_{year_of_interest}.csv',
                                         index_col=0)
erzeugerpreisindex = pd.read_csv('/Users/tomkaehler/Documents/Uni/BA/output/erzeugerpreisindex_extended.csv', index_col=0)

T_out = temperature_series_outside['temperature']
T_out.index = pd.to_datetime(T_out.index)

start_date = f'{year_of_interest}-01-01'
end_date = f'{year_of_interest}-12-31 23:00'

params = config['financial_params']

# Accessing individual parameters
fill_values = params['fill_values']
r = params['r']
nyears = params['nyears']
year = params['year']

cost_file = f"/Users/tomkaehler/Documents/Uni/BA/data/cost_files/costs_{params['year']}.csv"

cop_series = load_data_from_file(f'/Users/tomkaehler/Documents/Uni/BA/output/cop_series_{selected_system_id}_{year_of_interest}_supply:{supply_temp}.pkl')
all_technologies_dfs = load_data_from_file(
    f'/Users/tomkaehler/Documents/Uni/BA/output/all_technologies_dfs_{selected_system_id}_{year_of_interest}_supply:{supply_temp}_return:{return_temp}.pkl')
redispatch_wind_series = pd.read_csv(f'/Users/tomkaehler/Documents/Uni/BA/output/redispatch_wind_series_{selected_system_id}_{year_of_interest}',
                                     index_col=0)
redispatch_wind_series.index = pd.to_datetime(redispatch_wind_series.index)
redispatch_wind_series = redispatch_wind_series['0']


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
                cop_series[source_type][fluid][0].index = pd.to_datetime(cop_series[source_type][fluid][0].index)
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
            technology_df.at[name, 'M'] /= min_cop


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

n.add('Bus', 'gas')

if config.get('components_enabled', {}).get('redispatch wind power', False):
    # n.add('Bus', 'wind_redispatch')
    p_max_pu_redispatch = redispatch_wind_series / redispatch_wind_series.max()
    n.add("Generator",
          "windfarm",
          bus='power',
          p_nom_extendable=True,
          p_nom_max=redispatch_wind_series.max(),
          p_max_pu=p_max_pu_redispatch,
          marginal_cost=20,
          )


#    n.add('Line',
#          'Wind connect heat',
#          bus0='wind_redispatch',
#          bus1='power',
#          #p_nom_extendable=True,
#          length=3.4,
#          capital_cost=100 * 3.4,
#          x=0.1,
#          s_nom=100
#          )


n.add("Generator",
      "electricity market",
      bus="power",
      marginal_cost=electricity_price,
      p_nom_extendable=True,
      )

n.add('Generator',
      'gas market',
      bus='gas',
      marginal_cost=gas_price + co2_price * result.at['gas', 'CO2 intensity'],
      p_nom_extendable=True)

if "central water tank storage" in all_technologies_dfs:
    central_water_tank = all_technologies_dfs["central water tank storage"]
    n.add("Bus", "central_water_storage")

    n.add('Link',
          'central water tank charger',
          bus0='heat',
          bus1='central_water_storage',
          bus2='',
          efficiency=result.loc["water tank charger", "efficiency"],
          p_nom_extendable=True
          )

    n.add('Link',
          'central water tank discharger',
          bus0='central_water_storage',
          bus1='heat',
          bus2='',
          efficiency=result.loc["water tank discharger", "efficiency"],
          p_nom_extendable=True
          )
    # add hot water Storage
    n.madd("Store",
           central_water_tank.index,
           bus=["central_water_storage"],
           capital_cost=central_water_tank["fixed"],
           marginal_cost=central_water_tank['VOM'],
           e_cyclic=True,
           e_nom_extendable=True,
           )

if 'PTES' in all_technologies_dfs and supply_temp <= 95 + 273.15:
    PTES = all_technologies_dfs["PTES"]
    n.add('Bus', 'PTES')

    n.add('Link',
          'PTES charger',
          bus0='heat',
          bus1='PTES',
          bus2='',
          efficiency=result.loc["PTES charger", "efficiency"],
          p_nom_extendable=True
          )

    n.add('Link',
          'PTES discharger',
          bus0='PTES',
          bus1='heat',
          bus2='',
          efficiency=result.loc["PTES discharger", "efficiency"],
          p_nom_extendable=True
          )

    # add hot water Storage
    n.madd("Store",
           PTES.index,
           bus=["PTES"],
           capital_cost=PTES["fixed"],
           marginal_cost=PTES['VOM'],
           e_cyclic=True,
           e_nom_extendable=True,
           )

if config.get('components_enabled', {}).get('battery', False):
    n.add("Bus", "battery_storage")

    n.add(
        "Link",
        "battery charger",
        bus0='power',
        bus1='battery_storage',
        bus2='',
        # the efficiencies are "round trip efficiencies"
        efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
        capital_cost=(result.loc["battery inverter", "fixed"] * (erzeugerpreisindex.loc[
            'Ni-Cad-,Ni-Metallhydr-,Li-Ion-,Li-Polym-Akkus', project_year] /
            erzeugerpreisindex.loc['Ni-Cad-,Ni-Metallhydr-,Li-Ion-,Li-Polym-Akkus', '2020'])).round(3),
        marginal_cost=result.loc["battery inverter", "VOM"],
        p_nom_extendable=True
    )

    # add battery storage battery inverter
    n.add("Store",
          "battery store",
          bus="battery_storage",
          e_cyclic=True,
          capital_cost=(result.loc["battery storage", "fixed"] * (erzeugerpreisindex.loc[
            'Ni-Cad-,Ni-Metallhydr-,Li-Ion-,Li-Polym-Akkus', project_year] /
            erzeugerpreisindex.loc['Ni-Cad-,Ni-Metallhydr-,Li-Ion-,Li-Polym-Akkus', '2020'])).round(3),
          e_nom_extendable=True
          )

    n.add(
        "Link",
        "battery discharger",
        bus0='battery_storage',
        bus1='power',
        bus2='',
        efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
        p_nom_extendable=True
    )

# add H2 storage
n.add("Bus", "H2")


if "electrolysis AEC" in all_technologies_dfs:
    electrolysis_AEC = all_technologies_dfs["electrolysis AEC"]
    #if "H2" not in n.buses.index:  # Check if H2 bus is not already added
    #    n.add("Bus", "H2")
    n.madd("Link",
           electrolysis_AEC.index,
           bus0=["power"],
           bus1=["H2"],
           bus2=['Excess_heat'],
           capital_cost=electrolysis_AEC["fixed"],
           marginal_cost=electrolysis_AEC['VOM'],
           efficiency=electrolysis_AEC["efficiency"],
           efficiency2=electrolysis_AEC["efficiency-heat"],
           p_nom_extendable=True
           )

if "electrolysis PEMEC" in all_technologies_dfs:
    electrolysis_PEMEC = all_technologies_dfs["electrolysis PEMEC"]
    #if "H2" not in n.buses.index:  # Check if H2 bus is not already added
    #    n.add("Bus", "H2")
    n.madd("Link",
           electrolysis_PEMEC.index,
           bus0=["power"],
           bus1=["H2"],
           bus2=['Excess_heat'],
           capital_cost=electrolysis_PEMEC["fixed"],
           marginal_cost=electrolysis_PEMEC['VOM'],
           efficiency=electrolysis_PEMEC["efficiency"],
           efficiency2=electrolysis_PEMEC["efficiency-heat"],
           p_nom_extendable=True
           )

if config.get('components_enabled', {}).get('H2 store', False):
    n.add("Bus", "H2_storage")

    n.add(
        "Link",
        "H2 charger",
        bus0='H2',
        bus1='H2_storage',
        bus2='',
        # the efficiencies are "round trip efficiencies"
        efficiency=result.loc["hydrogen storage tank type 1 including compressor", "efficiency"] ** 0.5,
        p_nom_extendable=True
    )

    # add battery storage battery inverter
    n.add("Store",
          "H2 store",
          bus="H2_storage",
          e_cyclic=True,
          capital_cost=(result.loc["hydrogen storage tank type 1 including compressor", "fixed"] * (
          erzeugerpreisindex.loc['Behälter f. verdicht. od. verflüss. Gase,aus Eisen', project_year] /
          erzeugerpreisindex.loc['Behälter f. verdicht. od. verflüss. Gase,aus Eisen', '2020'])).round(3),
          e_nom_extendable=True
          )

    n.add(
        "Link",
        "H2 discharger",
        bus0='H2_storage',
        bus1='H2',
        bus2='',
        efficiency=result.loc["hydrogen storage tank type 1 including compressor", "efficiency"] ** 0.5,
        p_nom_extendable=True
    )

if "central H2 boiler" in all_technologies_dfs:
    central_H2_boiler = all_technologies_dfs["central H2 boiler"]
    n.madd("Link",
           central_H2_boiler.index,
           bus0=["H2"],
           bus1=["heat"],
           bus2=[''],
           p_nom_extendable=True,
           efficiency=central_H2_boiler["efficiency"],
           capital_cost=central_H2_boiler["fixed"],
           marginal_cost=central_H2_boiler["VOM"]
           )


"""""
Die Annahme stimmt soweit bus0 beschreibt mein input und jeglicher weiterer Bus outputs bzw. input wenn negativ. 
Die efficiency wird in das Verhältnis gesetzt zum bus0 
"""""

if "central excess-heat heat pump" in all_technologies_dfs:
    n.add('Bus', 'Excess_heat')
    eh_electrolysis_hp = all_technologies_dfs["central excess-heat heat pump"]
    cop_eh_hp = calculate_cop(eh_electrolysis_hp, cop_series)
    update_capacity(eh_electrolysis_hp, cop_eh_hp)

    n.madd('Link',
           eh_electrolysis_hp.index,
           bus0=['Excess_heat'],
           bus1=['heat'],
           bus2=['power'],
           capital_cost=eh_electrolysis_hp["fixed"],
           marginal_cost=eh_electrolysis_hp['VOM'],
           efficiency=(1 / (cop_eh_hp - 1)) * cop_eh_hp,
           efficiency2=-1 / (cop_eh_hp - 1),
           p_nom_extendable=True
           )

if "central sourced-water heat pump" in all_technologies_dfs:
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
           bus0=['power'],
           bus1=['heat'],
           bus2=['river_withdrawal'],
           capital_cost=river_hp["fixed"],
           marginal_cost=river_hp['VOM'],
           efficiency=1 * cop_river_hp,
           efficiency2=-1 * cop_river_hp / (cop_river_hp - 1),
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
           lake_hp.index,
           bus0=['river_withdrawal'],
           bus1=['heat'],
           bus2=['power'],
           capital_cost=lake_hp["fixed"],
           marginal_cost=lake_hp['VOM'],
           efficiency=(1 * (cop_lake_hp - 1)) / cop_lake_hp,
           efficiency2=-1 * (cop_lake_hp - 1),
           p_nom_extendable=True
           )

if "central air sourced heat pump" in all_technologies_dfs:
    central_air_hp = all_technologies_dfs["central air sourced heat pump"]
    cop_air_hp = calculate_cop(central_air_hp, cop_series)
    update_capacity(central_air_hp, cop_air_hp)

    n.madd("Link",
           central_air_hp.index,
           bus0=["power"],  # Assuming all heat pumps are connected to the same bus
           bus1=["heat"],  # Assuming all heat pumps supply to the same bus
           bus2=[''],
           p_nom_extendable=True,
           efficiency=cop_air_hp,
           capital_cost=central_air_hp["fixed"],
           marginal_cost=central_air_hp['VOM']
           )

if "central electric boiler" in all_technologies_dfs:
    central_electric_boiler = all_technologies_dfs["central electric boiler"]
    # add e-boiler
    n.madd("Link",
           central_electric_boiler.index,
           bus0=["power"],
           bus1=["heat"],
           bus2=[''],
           efficiency=central_electric_boiler["efficiency"],
           capital_cost=central_electric_boiler["fixed"],
           marginal_cost=central_electric_boiler["VOM"],
           p_nom_extendable=True
           )

if "central gas boiler" in all_technologies_dfs:
    central_gas_boiler = all_technologies_dfs['central gas boiler']
    n.madd('Link',
           central_gas_boiler.index,
           bus0=['gas'],
           bus1=['heat'],
           efficiency=central_gas_boiler["efficiency"],
           capital_cost=central_gas_boiler["fixed"],
           marginal_cost=central_gas_boiler["VOM"],
           p_nom_extendable=True,
           )


def separate_links_and_stores(all_technologies_dfs):
    # Initialize an empty list for links and stores
    links_list = []
    stores_list = []

    # Specify the names that identify a store
    store_names = ['PTES', 'central water tank storage']

    # Loop through the dictionary, checking the name of each DataFrame
    for name, df in all_technologies_dfs.items():
        if name in store_names:
            stores_list.append(df)
        else:
            links_list.append(df)

    # Use a try-except block or check length to avoid errors with pd.concat on empty lists
    if links_list:
        links_df = pd.concat(links_list)
    else:
        links_df = pd.DataFrame()
    if stores_list:
        stores_df = pd.concat(stores_list)
    else:
        stores_df = pd.DataFrame()

    return links_df, stores_df


links_df, stores_df = separate_links_and_stores(all_technologies_dfs)
# Concatenate both, ensuring empty DataFrames are handled correctly
all_components_df = pd.concat([links_df, stores_df]) if not (links_df.empty and stores_df.empty) else pd.DataFrame()


def custom_constraints(n, sns):
    n.model.add_variables(coords=[all_components_df.index], name="Link-build", binary=True)
    # Retrieve the binary variable
    binary_var = n.model["Link-build"]
    print(binary_var)

    def get_component_base_name(idx):
        return re.sub(r'_\d+$', '', idx)

    # Group indices by component base names
    def add_exclusive_selection_constraints():
        # Group indices by component base names within this function
        grouped_indices = all_components_df.groupby(all_components_df.index.map(get_component_base_name))

        for component_base, idx_group in grouped_indices:
            indices = idx_group.index
            if len(indices) > 1:  # Apply constraint if multiple segments exist
                # Create a list of constraints enforcing that the sum of binary vars for these indices <= 1
                constraints = {idx: binary_var[idx] for idx in indices}
                n.model.add_constraints(sum(constraints.values()) <= 1, name=f"exclusive_{component_base}")

    def link_constraint():
        if links_df.empty or not n.links.p_nom_extendable.any():
            print("Skipping constraint due to empty links_df or no extendable links.")
            return

        # Get the variable p_nom_opt for the heat pumps
        p_nom_opt = n.model["Link-p_nom"].loc[links_df.index]

        # by this way we can access the input and output per snapshot, so we are able to get partial load behavior
        # p_one_component = n.model["Link-p"].loc[sns, all_components_df.index]
        # print(p_one_component)

        p_nom_min = links_df["Start Capacity"].loc[links_df.index]

        m = links_df["M"].loc[links_df.index]

        for comp in links_df.index:
            # Apply the constraints
            n.model.add_constraints(p_nom_opt[comp] - binary_var[comp] * m[comp] <= 0,
                                    name=f"Link-p_nom-upperlimit-{comp}")
            n.model.add_constraints(p_nom_opt[comp] - binary_var[comp] * m[comp] >= p_nom_min[comp] - m[comp],
                                    name=f"Link-p_nom-lowerlimit-{comp}")

    def store_constraint():
        if stores_df.empty or not n.stores.e_nom_extendable.any():
            return

        # Get the variable p_nom_opt for the heat pumps
        e_nom_opt = n.model['Store-e_nom'].loc[stores_df.index]
        print(e_nom_opt)

        # by this way we can access the input and output per snapshot, so we are able to get partial load behavior
        # p_one_component = n.model["Link-p"].loc[sns, all_components_df.index]
        # print(p_one_component)

        e_nom_min = stores_df["Start Capacity"].loc[stores_df.index]
        print(e_nom_min)

        m = stores_df["M"].loc[stores_df.index]
        print(m)

        for comp in stores_df.index:
            # Apply the constraints
            n.model.add_constraints(e_nom_opt[comp] - binary_var[comp] * m[comp] <= 0,
                                    name=f"Link-p_nom-upperlimit-{comp}")
            n.model.add_constraints(e_nom_opt[comp] - binary_var[comp] * m[comp] >= e_nom_min[comp] - m[comp],
                                    name=f"Link-p_nom-lowerlimit-{comp}")

    add_exclusive_selection_constraints()
    link_constraint()
    store_constraint()


thh_series = pd.read_csv(f'/Users/tomkaehler/Documents/Uni/BA/output/thh_series_{selected_system_id}_{year_of_interest}.csv', index_col=0)
thh_series = thh_series['THH']
thh_series.index = pd.to_datetime(thh_series.index)

# add demand
n.add("Load",
      "heat_demand",
      bus="heat",
      p_set=thh_series)

logging.getLogger('pypsa').setLevel(logging.WARNING)
logging.getLogger('gurobipy').setLevel(logging.WARNING)

# Specify the solver options
solver_options = {
    'gurobi': {
        'LogToConsole': 0,
        'MIPFocus': 1, #for H2 storage
        'MIPGap': 0.03,
        # 'TimeLimit': 3600,
        'Heuristics': 0.1, #for H2 storage
        # 'Cuts': 1,
        # 'Threads': 8,
        # 'Presolve': 1,
        # 'NumericFocus': 3,       # Favour numeric stability over speed
        # 'FeasibilityTol': 1e-5,
        # 'OptimalityTol': 1e-5,
        # 'Method': 3,
        # 'NodeMethod': 2,
        # 'crossover': 0,
        # 'AggFill': 0,
        # 'PreDual': 0
    }
}

if 'build_opt' not in n.links.columns:
    n.links['build_opt'] = 0  # Initialize with a default value (e.g., 0)
if 'build_opt' not in n.stores.columns:
    n.stores['build_opt'] = 0

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
print(n.links.p_nom_opt)

# plot results (mal gucken was das kann und wofür mann das braucht)
# n.generators_t.p.plot()

# get statistics (mal gucken was das kann und wofür mann das braucht)
statistics = n.statistics().dropna()
#balance = n.statistics.energy_balance()
#print(balance)

n.export_to_csv_folder(
    f"/Users/tomkaehler/Documents/Uni/BA/output/results_{selected_system_id}_{year_of_interest}_supply:{supply_temp}_return:{return_temp}_elecprice_{electricity_price}_gasprice_{gas_price}_CO2price{co2_price}_run_{run}")
print("csv exported sucessfully")
#n.export_to_netcdf(
#    f"output/results_{selected_system_id}_{year_of_interest}_supply:{supply_temp}_return:{return_temp}_elecprice_{electricity_price}_gasprice_{gas_price}_CO2price{co2_price}_run_{run}.nc")

links = n.links
components = n.components

thh_sum = thh_series.sum()

total_system_cost = n.objective

print("Total System Cost:", total_system_cost)
waermegestehungskosten = total_system_cost / thh_sum
print("Wärmegestehungskosten:", waermegestehungskosten)

dict_summary = {
    "thh_sum": [thh_sum],
    "total_system_cost": [total_system_cost],
    "waermegestehungskosten": [waermegestehungskosten]
}

summary = pd.DataFrame(dict_summary)

# Save the DataFrame to a CSV file
summary.to_csv(f'/Users/tomkaehler/Documents/Uni/BA/output/_{selected_system_id}_{year_of_interest}_supply:{supply_temp}_return:{return_temp}_elecprice_{electricity_price}_gasprice_{gas_price}_CO2price{co2_price}_run_{run}.csv')
