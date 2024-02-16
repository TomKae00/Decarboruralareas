# import packages
import pypsa
import pandas as pd
import numpy as np
import yaml

from weather_data import temperature_series
from Wind_power import wind_series_for_pypsa
from linearizing_costs import all_technologies_dfs

# so nicht gut gelöst. Gucken dass diese Werte in def eingebracht werden und dann in einem Skript für die Parameter festgelegt werden
T_out = temperature_series
VLT = 70
RLT = 35
dT_HP = 4  # Abkühlung der Quelltemperatur
dT = VLT - 40
cp = 4186e-6
p = 998
e_nom_max_heat_storage = 100e6

size = e_nom_max_heat_storage / (dT * cp * p)

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

    costs = costs.fillna(params["fill_values"])  # what is the params dictionary?

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

        # Update component_df
        component_df['efficiency'] = efficiency
        component_df['VOM'] = VOM
        component_df['FOM'] = FOM
        component_df['lifetime'] = lifetime


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

# create PyPSA network
n = pypsa.Network()
# set time steps
n.set_snapshots(pd.date_range(start="2019-01-01", end="2019-12-31", freq="h"))
sns = n.snapshots

# add a bus/region for electricity
n.add("Bus", "power")

# add generator for wind generation
n.add("Generator",
      "windfarm",
      bus="power",
      capital_cost=result.loc["onwind", "investment"],
      marginal_cost=result.loc["onwind", "VOM"],
      p_nom_extendable=True,
      p_max_pu=wind_series_for_pypsa)
# p_max_pu=random_series)

# add a bus for district heating
n.add("Bus", "heat")
# n.add("Bus", "hot_water_storage")


# add hot water Storage
n.add("Store",
      "Waerme_speicher",
      bus="heat",
      capital_cost=result.loc["central water tank storage", "fixed"],
      e_cyclic=True,
      e_nom_extendable=True,
      standing_loss=0.01)  # für die standing_loss recherche betreiben. was angenommen werden kann -> Formel suchen in der Literatur Recheche

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
    p_nom_extendable=True,
    # marginal_cost=result.loc["battery inverter", "marginal_cost"],
)

# add battery storage battery inverter
n.add("Store",
      "battery store",
      bus="battery_storage",
      e_cyclic=True,
      capital_cost=result.loc["battery storage", "fixed"],
      efficiency=result.loc["battery storage", "efficiency"],
      # p_max_pu=?????, herausfinden für die Art von Batterie, die verwendet wird
      e_nom_extendable=True)

n.add(
    "Link",
    "battery discharger",
    bus0='battery_storage',
    bus1='power',
    carrier="battery discharger",
    efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
    p_nom_extendable=True,
    # marginal_cost=result.loc["battery inverter", "marginal_cost"],
)
# add H2 storage
n.add("Bus", "H2")
n.add('Bus', 'Excess_heat')
# n.add("Bus", "H2_storage")

# add electrolysis for H2 production
n.add("Link",
      "Elektrolyse",
      bus0="power",
      bus1="H2",
      bus2='Excess_heat',
      capital_cost=result.loc["electrolysis AEC", "fixed"],
      efficiency=result.at["electrolysis AEC", "efficiency"],
      efficiency2=result.at["electrolysis AEC", "efficiency-heat"],
      p_nom_extendable=True
      )

"""""
Die Annahme stimmt soweit bus0 beschreibt mein input und jeglicher weiterer Bus outputs bzw. input wenn negativ. 
Die efficiency wird in das Verhältnis gesetzt zum bus0 
"""""

# n.add('Link',
#      'excess_heat_HP',
#      bus0='Excess_heat',
#      bus1='heat',
#      bus2='power',
#      capital_cost=result.loc["electrolysis", "fixed"],
#      efficiency=(1/(cop-1))*cop,
#      efficiency2=-1/(cop-1),
#      p_nom_extendable=True
#      )

n.add("Store",
      "H2 Speicher",
      bus="H2",
      e_cyclic=True,
      capital_cost=result.loc["hydrogen storage tank type 1 including compressor", "fixed"],
      e_nom_extendable=True)

# add H2 boiler
n.add("Link",
      "H2 Kessel",
      bus0="H2",
      bus1="heat",
      p_nom_extendable=True,
      carrier="H2 turbine",
      efficiency=result.at["OCGT", "efficiency"],
      capital_cost=result.at["OCGT", "fixed"] * result.at["OCGT", "efficiency"])

# n.add("Link",
#      "Waermepumpe",
#      bus0="power",
#      bus1="heat",
#      efficiency=cop, #3
#      p_nom_extendable=True,
#      #p_min_pu=0.001,
#      capital_cost=result.loc["central air-sourced heat pump", "fixed"], #500
#      )


if "central air sourced heat pump" in all_technologies_dfs:
    central_air_heat_pump_df = all_technologies_dfs["central air sourced heat pump"]
n.madd("Link",
       central_air_heat_pump_df.index,
       bus0=["power"],  # Assuming all heat pumps are connected to the same bus
       bus1=["heat"],  # Assuming all heat pumps supply to the same bus
       p_nom_extendable=True,
       # p_nom_min=HP_data_processed["p_nom_min"],
       # p_nom_max=HP_data_processed["p_nom_max"],
       # p_min_pu=HP_data_processed["p_min_pu"],                               # Assuming the same for all heat pumps
       efficiency=3,  # Assuming the same efficiency for all; replace if varying
       capital_cost=central_air_heat_pump_df["fixed"]
       )

if "central electric boiler" in all_technologies_dfs:
    central_electric_boiler_df = all_technologies_dfs["central electric boiler"]
# add e-boiler
n.madd("Link",
       central_electric_boiler_df.index,
       bus0=["power"],
       bus1=["heat"],
       efficiency=central_electric_boiler_df["efficiency"],
       capital_cost=central_electric_boiler_df["fixed"],
       marginal_cost=central_electric_boiler_df["VOM"],
       p_nom_extendable=True)


def custom_constraints(network, snapshots):

    heat_pump_constraints(network, snapshots)

    # Example: Electric Boiler Constraints
    electric_boiler_constraints(network, snapshots)


def heat_pump_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[central_air_heat_pump_df.index], name="Link-build", binary=True)
    # Retrieve the binary variable
    binary_var = n.model["Link-build"]
    print(binary_var)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[central_air_heat_pump_df.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = central_air_heat_pump_df["Start Capacity"].loc[central_air_heat_pump_df.index]
    print(p_nom_min)

    m = 30

    for hp in central_air_heat_pump_df.index:
        n.model.add_constraints(p_nom_opt[hp] - binary_var[hp] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{hp}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[hp] - binary_var[hp] * m >= p_nom_min[hp] - m,
                                name=f"Link-p_nom-lowerlimit-{hp}")



def electric_boiler_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[central_electric_boiler_df.index], name="Link-build2", binary=True)
    # Retrieve the binary variable
    binary_var2 = n.model["Link-build2"]
    print(binary_var2)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[central_electric_boiler_df.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = central_electric_boiler_df["Start Capacity"].loc[central_electric_boiler_df.index]
    print(p_nom_min)

    m = 30

    for eb in central_electric_boiler_df.index:
        n.model.add_constraints(p_nom_opt[eb] - binary_var2[eb] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{eb}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[eb] - binary_var2[eb] * m >= p_nom_min[eb] - m,
                                name=f"Link-p_nom-lowerlimit-{eb}")


# Load Heat Demand from CSV
# Falls Zeitformat nicht erkannt wird, dann anpassen mit , format='%Y-%m-%d' Achten auf die Buchstaben nachgucken unter python dates
heat_demand_data = pd.read_csv('data/heat_demand2.0.csv', sep=';', parse_dates=['timestamp'], date_format=date_input)
heat_demand_series = heat_demand_data.set_index('timestamp')['heat_demand'] / 1000
heat_demand_sum = heat_demand_series.sum()
# add demand
n.add("Load",
      "heat_demand",
      bus="heat",
      p_set=heat_demand_series)

# Specify the solver options
solver_options = {
    'gurobi': {
        'BarHomogeneous': 1  # Setting BarHomogeneous parameter for Gurobi
    }
}
c = 'link'  # Replace with the actual component name

if 'build_opt' not in n.links.columns:
    n.links['build_opt'] = 0  # Initialize with a default value (e.g., 0)

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

total_system_cost = n.objective
print("Total System Cost:", total_system_cost)
print(n.investment_period_weightings)
waermegestehungskosten = total_system_cost / heat_demand_sum
print("Wärmegestehungskosten:", waermegestehungskosten)
print(heat_demand_sum)

# Netzverlsute intgrieren
