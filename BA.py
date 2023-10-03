# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import packages
import pypsa
import pandas as pd
import numpy as np

# create PyPSA network
n = pypsa.Network()
# set time steps
n.set_snapshots(pd.date_range(start="2023-01-01", end="2023-01-31", freq="H"))
# gggg
# Synchronisiert

# add a bus/region for electricity
n.add("Bus", "windstrom")
# capacity factor of the wind farm (random values)
random_series = pd.Series(np.random.rand(len(n.snapshots)), index=n.snapshots)

# add generator for wind generation
n.add("Generator", "windfarm", bus="windstrom",
      capital_cost=1000, marginal_cost=2, p_nom_extendable=True,
      p_max_pu=random_series)


# add a bus for district heating
n.add("Bus", "fernwaerme")
n.add("Store",
      "Fernwaerme speicher",
      bus="fernwaerme",
      capital_cost=10,
      e_nom_extendable=True)

# add demand
n.add("Load",
      "Fernwaerme Nachfrage",
      bus="fernwaerme",
      p_set=10)

# add heat pump
# COP random
COP = pd.Series(4*np.random.rand(len(n.snapshots)), index=n.snapshots)
n.add("Link",
      "Waermepumpe",
      bus0="windstrom",
      bus1="fernwaerme",
      efficiency=COP,
      p_nom_extendable=True)

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
