# import packages
import pypsa
import pandas as pd
import numpy as np
import yaml

from weather_data import temperature_series
from Wind_power import wind_series_for_pypsa
from linearizing_costs import all_technologies_dfs
from heat_demand import thh_series
from cost_functions import cop_series

# so nicht gut gelöst. Gucken dass diese Werte in def eingebracht werden und dann in einem Skript für die Parameter festgelegt werden
T_out = temperature_series
VLT = 70
RLT = 35
dT_HP = 4  # Abkühlung der Quelltemperatur
dT = VLT - 40

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
        efficiency = result.at[technology_name, 'efficiency'] if 'efficiency' in result.columns else None
        VOM = result.at[technology_name, 'VOM'] if 'VOM' in result.columns else None
        FOM = result.at[technology_name, 'FOM'] if 'FOM' in result.columns else None
        lifetime = result.at[
            technology_name, 'lifetime'] if 'lifetime' in result.columns else 20  # Default to 20 if not specified
        efficiency_heat = result.at[technology_name, 'efficiency-heat'] if 'efficiency-heat' in result.columns else None

        # Update component_df
        component_df['efficiency'] = efficiency
        component_df['VOM'] = VOM
        component_df['FOM'] = FOM
        component_df['lifetime'] = lifetime
        component_df['efficiency-heat'] = efficiency_heat


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

date_input = "%Y-%m-%d %H:%M:%S"  # bis jetzt unnötig, aber noch übernehmen damit Daten einfacher ausgelesen werden können

"""""
At this point the creation of the network starts. 
"""""

# create PyPSA network
n = pypsa.Network()
# set time steps
n.set_snapshots(pd.date_range(start="2019-01-01", end="2019-12-31 23:00", freq="h"))
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
      p_nom_extandable=True
      )

n.add('Link',
      'water tank discharger',
      bus0='hot water storage',
      bus1='heat',
      efficiency=result.loc["water tank discharger", "efficiency"],
      p_nom_extandable=True
      )

# add hot water Storage
n.add("Store",
      "water tank",
      bus="hot water storage",
      capital_cost=result.loc["central water tank storage", "fixed"],
      e_cyclic=True,
      e_nom_extendable=True,
      standing_loss=0.2/24)  # für die standing_loss recherche betreiben. was angenommen werden kann -> Formel suchen in der Literatur Recheche

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
cop_eh_hp = pd.DataFrame(index=pd.date_range(start="2019-01-01", end="2019-12-31 23:00", freq="h"))
for name, row in eh_electrolysis_hp.iterrows():
    parts = row['cop_series'].split('_')
    if len(parts) >= 2:
        source_type = parts[0]
        fluid = '_'.join(parts[1:])  # Handling fluid names with '_'

        if source_type in cop_series and fluid in cop_series[source_type]:
            series_list = cop_series[source_type][fluid]['COP']
            if series_list and series_list[0].index.equals(cop_eh_hp.index):
                cop_eh_hp[name] = series_list[0]

for name, row in eh_electrolysis_hp.iterrows():
    if name in cop_eh_hp.columns:
        # Extract non-zero COP values and calculate the minimum COP
        min_cop = cop_eh_hp[cop_eh_hp[name] > 0][name].min()

        # Calculate and update the Start and End capacities in MW_el
        eh_electrolysis_hp.at[name, 'Start Capacity MW_el'] = row['Start Capacity'] / min_cop
        eh_electrolysis_hp.at[name, 'End Capacity MW_el'] = row['End Capacity'] / min_cop

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
cop_river_hp = pd.DataFrame(index=pd.date_range(start="2019-01-01", end="2019-12-31 23:00", freq="h"))
for name, row in river_hp.iterrows():
    parts = row['cop_series'].split('_')
    if len(parts) >= 2:
        source_type = parts[0]
        fluid = '_'.join(parts[1:])  # Handling fluid names with '_'

        if source_type in cop_series and fluid in cop_series[source_type]:
            series_list = cop_series[source_type][fluid]['COP']
            if series_list and series_list[0].index.equals(cop_river_hp.index):
                cop_river_hp[name] = series_list[0]

for name, row in river_hp.iterrows():
    if name in cop_river_hp.columns:
        # Extract non-zero COP values and calculate the minimum COP
        min_cop = cop_river_hp[cop_river_hp[name] > 0][name].min()

        # Calculate and update the Start and End capacities in MW_el
        river_hp.at[name, 'Start Capacity MW_el'] = row['Start Capacity'] / min_cop
        river_hp.at[name, 'End Capacity MW_el'] = row['End Capacity'] / min_cop

n.add('Bus',
      'river_withdrawal'
      )

n.add('Generator',
      'river_potential',
      bus='river_withdrawal',
      p_nom_max=50, # ersetzen mit dem Wert, der noch berechnet werden muss
      p_nom_extandable=True
      )

n.madd('Link',
       river_hp.index,
       bus0=['river_potential'],
       bus1=['heat'],
       bus2=['power'],
       capital_cost=river_hp["fixed"],
       efficiency=(1 / (cop_river_hp - 1)) * cop_river_hp,
       efficiency2=-1 / (cop_river_hp - 1),
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

cop_air_hp = pd.DataFrame(index=pd.date_range(start="2019-01-01", end="2019-12-31 23:00", freq="h"))
for name, row in central_air_hp.iterrows():
    parts = row['cop_series'].split('_')
    if len(parts) >= 2:
        source_type = parts[0]
        fluid = '_'.join(parts[1:])  # Handling fluid names with '_'

        if source_type in cop_series and fluid in cop_series[source_type]:
            series_list = cop_series[source_type][fluid]['COP']
            if series_list and series_list[0].index.equals(cop_air_hp.index):
                cop_air_hp[name] = series_list[0]

for name, row in central_air_hp.iterrows():
    # Assuming the name of the heat pump directly matches the column in cop_air_hp DataFrame
    if name in cop_air_hp.columns:
        # Extract non-zero COP values and calculate the minimum COP
        min_cop = cop_air_hp[cop_air_hp[name] > 0][name].min()

        # Calculate and update the Start and End capacities in MW_el
        central_air_hp.at[name, 'Start Capacity MW_el'] = row['Start Capacity'] / min_cop
        central_air_hp.at[name, 'End Capacity MW_el'] = row['End Capacity'] / min_cop



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


def custom_constraints(network, snapshots):

    air_heat_pump_constraints(network, snapshots)
    eh_heat_pump_constraints(network, snapshots)
    electric_boiler_constraints(network, snapshots)
    H2_boiler_constraints(network, snapshots)
    electrolysis_AEC_constraints(network, snapshots)


def air_heat_pump_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[central_air_hp.index], name="Link-build", binary=True)
    # Retrieve the binary variable
    binary_var = n.model["Link-build"]
    print(binary_var)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[central_air_hp.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = central_air_hp["Start Capacity MW_el"].loc[central_air_hp.index]
    print(p_nom_min)

    m = 8

    for hp in central_air_hp.index:
        n.model.add_constraints(p_nom_opt[hp] - binary_var[hp] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{hp}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[hp] - binary_var[hp] * m >= p_nom_min[hp] - m,
                                name=f"Link-p_nom-lowerlimit-{hp}")

def eh_heat_pump_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[eh_electrolysis_hp.index], name="Link-build1", binary=True)
    # Retrieve the binary variable
    binary_var1 = n.model["Link-build1"]
    print(binary_var1)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[eh_electrolysis_hp.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = eh_electrolysis_hp["Start Capacity MW_el"].loc[eh_electrolysis_hp.index]
    print(p_nom_min)

    m = 30

    for eh in eh_electrolysis_hp.index:
        n.model.add_constraints(p_nom_opt[eh] - binary_var1[eh] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{eh}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[eh] - binary_var1[eh] * m >= p_nom_min[eh] - m,
                                name=f"Link-p_nom-lowerlimit-{eh}")


def electric_boiler_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[central_electric_boiler.index], name="Link-build2", binary=True)
    # Retrieve the binary variable
    binary_var2 = n.model["Link-build2"]
    print(binary_var2)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[central_electric_boiler.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = central_electric_boiler["Start Capacity"].loc[central_electric_boiler.index]
    print(p_nom_min)

    m = 30

    for eb in central_electric_boiler.index:
        n.model.add_constraints(p_nom_opt[eb] - binary_var2[eb] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{eb}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[eb] - binary_var2[eb] * m >= p_nom_min[eb] - m,
                                name=f"Link-p_nom-lowerlimit-{eb}")


def H2_boiler_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[central_H2_boiler.index], name="Link-build3", binary=True)
    # Retrieve the binary variable
    binary_var3 = n.model["Link-build3"]
    print(binary_var3)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[central_H2_boiler.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = central_H2_boiler["Start Capacity"].loc[central_H2_boiler.index]
    print(p_nom_min)

    m = 30

    for H2 in central_H2_boiler.index:
        n.model.add_constraints(p_nom_opt[H2] - binary_var3[H2] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{H2}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[H2] - binary_var3[H2] * m >= p_nom_min[H2] - m,
                                name=f"Link-p_nom-lowerlimit-{H2}")


def electrolysis_AEC_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[electrolysis_AEC.index], name="Link-build4", binary=True)
    # Retrieve the binary variable
    binary_var4 = n.model["Link-build4"]
    print(binary_var4)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[electrolysis_AEC.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = electrolysis_AEC["Start Capacity"].loc[electrolysis_AEC.index]
    print(p_nom_min)

    m = 1500

    for AEC in electrolysis_AEC.index:
        n.model.add_constraints(p_nom_opt[AEC] - binary_var4[AEC] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{AEC}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[AEC] - binary_var4[AEC] * m >= p_nom_min[AEC] - m,
                                name=f"Link-p_nom-lowerlimit-{AEC}")


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

if 'build1_opt' not in n.links.columns:
    n.links['build1_opt'] = 0

if 'build2_opt' not in n.links.columns:
    n.links['build2_opt'] = 0

if 'build3_opt' not in n.links.columns:
    n.links['build3_opt'] = 0

if 'build4_opt' not in n.links.columns:
    n.links['build4_opt'] = 0

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
