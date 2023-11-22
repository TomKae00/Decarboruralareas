# import packages
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#noch kein Plan, was das macht
#from urllib.request import urlretrieve

date_input = "%Y-%m-%d %H:%M:%S"

# create new .py for the calculation of the COP. If not too long add to main script
#VLT = 90
#RLT = 75

# cost function heat pump
def capital_cost_HP(p_nom):
      cost = 3 * p_nom**2 + 2 * p_nom
      return cost

plt.style.use("bmh")

# create PyPSA network
n = pypsa.Network()
# set time steps
n.set_snapshots(pd.date_range(start="2023-01-01", end="2023-01-31", freq="H"))

# add the costs of 2020
costs_20 = pd.read_csv("data/costs_2020.csv")

# add a bus/region for electricity
n.add("Bus", "windstrom")
# capacity factor of the wind farm (random values)
#random_series = pd.Series(np.random.rand(len(n.snapshots)), index=n.snapshots)

#das geht auch anders! gucken, wie man direkt eine Serie reinlädt, wahrscheinlich über index = snapshots
wind_power_data = pd.read_csv('data/wind_power.csv', parse_dates=['timestamp'], date_format=date_input)
wind_power_series = wind_power_data.set_index('timestamp')['wind_power']

# add generator for wind generation
n.add("Generator", "windfarm", bus="windstrom",
      capital_cost=1000,
      marginal_cost=2,
      p_nom_extendable=True,
      p_set=wind_power_series)
      #p_max_pu=random_series)

# add a bus for district heating
n.add("Bus", "Waerme")

# add a bus for H2
n.add("Bus","H2")


#Storages

# add hot water Storage
n.add("Store",
      "Waerme speicher",
      bus="Waerme",
      capital_cost=10,
      e_nom_max=1000,
      e_nom_min=500,
      e_nom_extendable=True,
      standing_loss=0.01)
# add battery storage
n.add("Store",
      "Batterie Speicher",
      bus="windstrom",
      capital_cost=9,
      e_nom_extendable=True)
# add H2 storage
n.add("Store",
      "H2 Speicher",
      bus="H2",
      capital_cost=costs_20.iloc[159, 2],
      e_nom_extendable=True)

# add electrolysis for H2 production
n.add("Link",
      "Elektrolyse",
      bus0="windstrom",
      bus1="H2",
      p_nom_extendable=True)

# add H2 boiler
n.add("Link",
      "H2 Kessel",
      bus0="H2",
      bus1="Waerme",
      efficiency=0.94,
      p_nom_extendable=True)


# add heat conversion technologies

# add heat pump
# calculating COP
capital_costs = capital_cost_HP(n.generators.p_nom)
COP = pd.Series(4*np.random.rand(len(n.snapshots)), index=n.snapshots)
n.add("Link",
      "Waermepumpe",
      bus0="windstrom",
      bus1="Waerme",
      efficiency=COP,
      p_nom_extendable=True,
      capital_cost=capital_cost_HP)

# add e-boiler
n.add("Link",
      "E-Kessel",
      bus0="windstrom",
      bus1="Waerme",
      efficiency=0.95,
      p_nom_extendable=True)

# Load Heat Demand from CSV
#Falls Zeitformat nicht erkannt wird, dann anpassen mit , format='%Y-%m-%d' Achten auf die Buchstaben nachgucken unter python dates
heat_demand_data = pd.read_csv('data/heat_demand.csv', parse_dates=['timestamp'], date_format=date_input)
heat_demand_series = heat_demand_data.set_index('timestamp')['heat_demand']
# add demand
n.add("Load",
      "Fernwaerme Nachfrage",
      bus="Waerme",
      p_set=heat_demand_series)

n.optimize()

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
#n.statistics()
#n.statistics.energy_balance()

n.export_to_csv_folder("results")
print("csv exported sucessfully")

#more to come