# import packages
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from weather_data import temperature_series

#im optimize code direkt integrieren, dass eine dynamische Auswahl der Kosten möglich ist.

# so nicht gut gelöst. Gucken dass diese Werte in def eingebracht werden und dann in einem Skript für die Parameter festgelegt werden
T_out = temperature_series
VLT = 70
RLT = 35
dT_HP = 4 # Abkühlung der Quelltemperatur
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

#n.snapshot_weightings // nochmal gucken was ditte kann

cost_file = f"data/costs_{params['year']}.csv"

# Funktion für die Berechnung der Kosten, übernommen aus prepare_sector_network.py
# Damit dies funktioniert muss noch noch calculate annuity aus add electricity übernommen werden


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

    costs = costs.fillna(params["fill_values"]) # what is the params dictionary?

    def annuity_factor(v):
        return calculate_annuity(v["lifetime"], params["r"]) + v["FOM"] / 100

    costs["annuity_factor"] = costs.apply(annuity_factor, axis=1)

    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * nyears for i, v in costs.iterrows()
    ] # .itterows macht genau das gleiche wie .apply

    return costs


result = prepare_costs(cost_file, params, params["nyears"])
print(result)

date_input = "%Y-%m-%d %H:%M:%S" #bis jetzt unnötig, aber noch übernehmen damit Daten einfacher ausgelesen werden können

plt.style.use("bmh") #an falscher Stelle

# create PyPSA network
n = pypsa.Network()
# set time steps
n.set_snapshots(pd.date_range(start="2019-01-01", end="2019-12-31", freq="h"))

# add a bus/region for electricity
n.add("Bus", "windstrom")

# capacity factor of the wind farm
#das geht auch anders! gucken, wie man direkt eine Serie reinlädt, wahrscheinlich über index = snapshots
wind_power_data = pd.read_csv('data/wind_power2.0.csv', sep=';', parse_dates=['timestamp'], date_format=date_input)
wind_power_series = wind_power_data.set_index('timestamp')['wind_power']/1000

# add generator for wind generation
n.add("Generator",
      "windfarm",
      bus="windstrom",
      capital_cost=100,
      marginal_cost=140,
      p_nom_extendable=True,
      p_set=wind_power_series)
      #p_max_pu=random_series)

# add a bus for district heating
n.add("Bus", "Waerme")
n.add("Bus","H2")
n.add("Bus", "battery_storage")
n.add("Bus", "H2_storage")
n.add("Bus", "hot_water_storage")


def calculate_heat_storage_capacity(dT):
    # Spezifische Wärmekapazität von Wasser in J/(kg*K)
    return dT * cp * size


# Berechne die gespeicherte Wärmemenge
stored_heat = calculate_heat_storage_capacity(dT)

# add hot water Storage
n.add("Store",
      "Waerme_speicher",
      bus="Waerme",
      capital_cost=result.loc["central water tank storage", "fixed"],
      e_cyclic=True,
      #e_nom_max=150,
      #e_nom_min=0,
      e_nom_extendable=True,
      #e_max_pu=1.1,
      standing_loss=0.01) #für die standing_loss recherche betreiben. was angenommen werden kann -> Formel suchen in der Literatur Recheche

n.add(
    "Link",
    "battery charger",
    bus0='wind_power',
    bus1='battery_storage',
    carrier="battery charger",
    # the efficiencies are "round trip efficiencies"
    efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
    capital_cost=result.loc["battery inverter", "fixed"],
    p_nom_extendable=True,
    #marginal_cost=result.loc["battery inverter", "marginal_cost"],
)

# add battery storage battery inverter
n.add("Store",
      "battery store",
      bus="battery_storage",
      e_cyclic=True,
      capital_cost=result.loc["battery storage", "fixed"],
      efficiency=result.loc["battery storage", "efficiency"],
      #p_max_pu=?????, herausfinden für die Art von Batterie, die verwendet wird
      e_nom_extendable=True)

n.add(
    "Link",
    "battery discharger",
    bus0='battery_storage',
    bus1='wind_power',
    carrier="battery discharger",
    efficiency=result.loc["battery inverter", "efficiency"] ** 0.5,
    p_nom_extendable=True,
    #marginal_cost=result.loc["battery inverter", "marginal_cost"],
)
# add H2 storage

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
      bus1="Waerme",
      p_nom_extendable=True,
      carrier="H2 turbine",
      efficiency=result.at["OCGT", "efficiency"],
      capital_cost=result.at["OCGT", "fixed"] * result.at["OCGT", "efficiency"])

HP_data = f"data/HP_data_{params['year']}.csv"


def costs_HP(HP_data, nyears, VLT):
    costs_HP = pd.read_csv(HP_data)

    costs_HP['p_nom_max'] = costs_HP['p_nom_max'].apply(
        lambda x: float('inf') if isinstance(x, str) and x.strip("'") == 'inf' else float(x))

    costs_HP = costs_HP[(costs_HP['VLT'] >= VLT)]

    # correct units to MW and EUR
    costs_HP['investment'] *= 1e3

    def annuity_factor_HP(v):
        return calculate_annuity(v["lifetime"], params["r"]) + v["FOM"] / 100

    costs_HP["annuity_factor"] = costs_HP.apply(annuity_factor_HP, axis=1)

    costs_HP["fixed"] = [
        annuity_factor_HP(v) * v["investment"] * nyears for i, v in costs_HP.iterrows()
    ] # .itterows macht genau das gleiche wie .apply

    return costs_HP


HP_data_processed = costs_HP(HP_data, params["nyears"], VLT)
new_index = [f"heat_pump_{i+1}" for i in range(len(HP_data_processed))]
HP_data_processed.insert(0, 'name', new_index, True)
HP_data_processed = HP_data_processed.set_index('name')


#def calculate_cop(T_out, VLT):
#    dT = VLT - T_out
#    COP = 6.08 - 0.09 * dT + 0.0005 * dT**2
#
#    # Überprüfen Sie die Bedingung für jedes Element in der Series
#    condition = T_out <= -5 + 273.15
#
#    # Setzen Sie den COP auf 0 für die Elemente, die die Bedingung erfüllen
#    if isinstance(COP, pd.Series):
#        COP[condition] = 0
#    elif COP > 0 and condition:
#        COP = 0
#
#    return COP
#
#
#cop = calculate_cop(T_out, VLT)
#print(cop)

t_lm_sink = (VLT-RLT)/(np.log((VLT+273.15)/(RLT+273.15)))
t_lm_source = (T_out-dT_HP)/(np.log((T_out+273.15)/(dT_HP+273.15)))
n_lorenz = 0.5

def calculate_cop(T_out, t_lm_sink, t_lm_source, n_lorenz):
    # Calculate COP based on the Lorenz cycle
    COP_lorenz = t_lm_sink / (t_lm_sink - t_lm_source)
    COP = COP_lorenz * n_lorenz

    # Condition to check if T_out is below -5 degrees Celsius
    condition = T_out <= -5

    # Handle both Series and scalar values of T_out
    if isinstance(T_out, pd.Series):
        COP[condition] = 0
    elif condition:
        COP = 0

    return COP


cop = calculate_cop(T_out, t_lm_sink, t_lm_source, n_lorenz)

# add heat pump
# use central air line 496
# calculating COP

#n.add("Link",
#      "Waermepumpe",
#      bus0="windstrom",
#      bus1="Waerme",
#      efficiency=cop, #3
#      p_nom_extendable=True,
#      #p_min_pu=0.001,
#      capital_cost=result.loc["central air-sourced heat pump", "fixed"], #500
#      )

n.madd("Link",
       HP_data_processed.index,
       bus0=["windstrom"],  # Assuming all heat pumps are connected to the same bus
       bus1=["Waerme"],    # Assuming all heat pumps supply to the same bus
       p_nom_extendable=True,
       #p_min_pu=HP_data_processed["p_min_pu"],                               # Assuming the same for all heat pumps
       efficiency=4,                            # Assuming the same efficiency for all; replace if varying
       capital_cost=HP_data_processed["fixed"]
       )

def heat_pump_constraints(n, sns):
    if not n.links.p_nom_extendable.any():
        return

    n.model.add_variables(coords=[HP_data_processed.index], name="Link-build", binary=True)
    # Retrieve the binary variable
    binary_var = n.model["Link-build"]
    print(binary_var)

    # Get the variable p_nom_opt for the heat pumps
    p_nom_opt = n.model["Link-p_nom"].loc[HP_data_processed.index]
    print(p_nom_opt)

    # Read p_nom_min from your processed data
    p_nom_min = HP_data_processed["p_nom_min"].loc[HP_data_processed.index]
    print(p_nom_min)

    m = 50

    for hp in HP_data_processed.index:
        n.model.add_constraints(p_nom_opt[hp] - binary_var[hp] * m <= 0,
                                name=f"Link-p_nom-upperlimit-{hp}")

        # Constraint 2: If binary_var is 0, p_nom_opt_pump must be 0
        n.model.add_constraints(p_nom_opt[hp] - binary_var[hp] * m >= p_nom_min[hp] - m,
                                name=f"Link-p_nom-lowerlimit-{hp}")

# add e-boiler
n.add("Link",
      "E-Kessel",
      bus0="windstrom",
      bus1="Waerme",
      efficiency=result.loc["electric boiler steam", "efficiency"],
      capital_cost=result.loc["electric boiler steam", "fixed"],
      marginal_cost=result.loc["electric boiler steam", "VOM"],
      p_nom_extendable=True)

# Load Heat Demand from CSV
#Falls Zeitformat nicht erkannt wird, dann anpassen mit , format='%Y-%m-%d' Achten auf die Buchstaben nachgucken unter python dates
heat_demand_data = pd.read_csv('data/heat_demand2.0.csv', sep=';', parse_dates=['timestamp'], date_format=date_input)
heat_demand_series = heat_demand_data.set_index('timestamp')['heat_demand']/1000
heat_demand_sum = heat_demand_series.sum()
# add demand
n.add("Load",
      "Fernwaerme Nachfrage",
      bus="Waerme",
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


n.optimize(n.snapshots, extra_functionality=heat_pump_constraints,
               solver_name="gurobi", solver_options=solver_options)
print("Optimization completed successfully")

binary_results = n.model["Link-build"].solution.to_pandas()


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
#n.generators_t.p.plot()
#n.plot()

# get statistics (mal gucken was das kann und wofür mann das braucht)
statistics = n.statistics().dropna()
curtailment = n.statistics.curtailment() #nicht klar was dies macht

n.statistics.energy_balance()

n.export_to_csv_folder("results")
print("csv exported sucessfully")

links = n.links

total_system_cost = n.objective
print("Total System Cost:", total_system_cost)
print(n.investment_period_weightings)
wärmegestehungskosten = total_system_cost/heat_demand_sum
print("Wärmegestehungskosten:", wärmegestehungskosten)
print(heat_demand_sum)


#more to com

#Netzverlsute intgrieren

#Consider using the
#homogeneous algorithm (through parameter 'BarHomogeneous')