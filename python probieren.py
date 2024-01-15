import pypsa, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.testing import assert_almost_equal, assert_array_almost_equal

def replace_su(network, su_to_replace):
    """Replace the storage unit su_to_replace with a bus for the energy
    carrier, two links for the conversion of the energy carrier to and from electricity,
    a store to keep track of the depletion of the energy carrier and its
    CO2 emissions, and a variable generator for the storage inflow.

    Because the energy size and power size are linked in the storage unit by the max_hours,
    extra functionality must be added to the LOPF to implement this constraint."""

    su = network.storage_units.loc[su_to_replace]

    bus_name = "{} {}".format(su["bus"], su["carrier"])
    link_1_name = "{} converter {} to AC".format(su_to_replace, su["carrier"])
    link_2_name = "{} converter AC to {}".format(su_to_replace, su["carrier"])
    store_name = "{} store {}".format(su_to_replace, su["carrier"])
    gen_name = "{} inflow".format(su_to_replace)

    network.add("Bus", bus_name, carrier=su["carrier"])

    # dispatch link
    network.add(
        "Link",
        link_1_name,
        bus0=bus_name,
        bus1=su["bus"],
        capital_cost=su["capital_cost"] * su["efficiency_dispatch"],
        p_nom=su["p_nom"] / su["efficiency_dispatch"],
        p_nom_extendable=su["p_nom_extendable"],
        p_nom_max=su["p_nom_max"] / su["efficiency_dispatch"],
        p_nom_min=su["p_nom_min"] / su["efficiency_dispatch"],
        p_max_pu=su["p_max_pu"],
        marginal_cost=su["marginal_cost"] * su["efficiency_dispatch"],
        efficiency=su["efficiency_dispatch"],
    )

    # store link
    network.add(
        "Link",
        link_2_name,
        bus0=su["bus"],
        bus1=bus_name,
        p_nom=su["p_nom"],
        p_nom_extendable=su["p_nom_extendable"],
        p_nom_max=su["p_nom_max"],
        p_nom_min=su["p_nom_min"],
        p_max_pu=-su["p_min_pu"],
        efficiency=su["efficiency_store"],
    )

    if (
        su_to_replace in network.storage_units_t.state_of_charge_set.columns
        and (
            ~pd.isnull(network.storage_units_t.state_of_charge_set[su_to_replace])
        ).any()
    ):
        e_max_pu = pd.Series(data=1.0, index=network.snapshots)
        e_min_pu = pd.Series(data=0.0, index=network.snapshots)
        non_null = ~pd.isnull(
            network.storage_units_t.state_of_charge_set[su_to_replace]
        )
        e_max_pu[non_null] = network.storage_units_t.state_of_charge_set[su_to_replace][
            non_null
        ]
        e_min_pu[non_null] = network.storage_units_t.state_of_charge_set[su_to_replace][
            non_null
        ]
    else:
        e_max_pu = 1.0
        e_min_pu = 0.0

    network.add(
        "Store",
        store_name,
        bus=bus_name,
        e_nom=su["p_nom"] * su["max_hours"],
        e_nom_min=su["p_nom_min"] / su["efficiency_dispatch"] * su["max_hours"],
        e_nom_max=su["p_nom_max"] / su["efficiency_dispatch"] * su["max_hours"],
        e_nom_extendable=su["p_nom_extendable"],
        e_max_pu=e_max_pu,
        e_min_pu=e_min_pu,
        standing_loss=su["standing_loss"],
        e_cyclic=su["cyclic_state_of_charge"],
        e_initial=su["state_of_charge_initial"],
    )

    network.add("Carrier", "rain", co2_emissions=0.0)

    # inflow from a variable generator, which can be curtailed (i.e. spilled)
    inflow_max = network.storage_units_t.inflow[su_to_replace].max()

    if inflow_max == 0.0:
        inflow_pu = 0.0
    else:
        inflow_pu = network.storage_units_t.inflow[su_to_replace] / inflow_max

    network.add(
        "Generator",
        gen_name,
        bus=bus_name,
        carrier="rain",
        p_nom=inflow_max,
        p_max_pu=inflow_pu,
    )

    if su["p_nom_extendable"]:
        ratio2 = su["max_hours"]
        ratio1 = ratio2 * su["efficiency_dispatch"]

        def extra_functionality(network, snapshots):
            model = network.model
            model.add_constraints(
                model["Store-e_nom"][store_name]
                - model["Link-p_nom"][link_1_name] * ratio1
                == 0,
                name="store_fix_1",
            )
            model.add_constraints(
                model["Store-e_nom"][store_name]
                - model["Link-p_nom"][link_2_name] * ratio2
                == 0,
                name="store_fix_2",
            )

    else:
        extra_functionality = None

    network.remove("StorageUnit", su_to_replace)

    return bus_name, link_1_name, link_2_name, store_name, gen_name, extra_functionality

network_r = pypsa.examples.storage_hvdc(from_master=True)
network_r.optimize()

network = pypsa.examples.storage_hvdc(from_master=True)

su_to_replace = "Storage 0"

(
    bus_name,
    link_1_name,
    link_2_name,
    store_name,
    gen_name,
    extra_functionality,
) = replace_su(network, su_to_replace)
network.optimize(extra_functionality=extra_functionality)

assert_almost_equal(network_r.objective, network.objective, decimal=2)
assert_array_almost_equal(
    network_r.storage_units_t.state_of_charge[su_to_replace],
    network.stores_t.e[store_name],
)
assert_array_almost_equal(
    network_r.storage_units_t.p[su_to_replace],
    -network.links_t.p1[link_1_name] - network.links_t.p0[link_2_name],
)

# check optimised size
assert_array_almost_equal(
    network_r.storage_units.at[su_to_replace, "p_nom_opt"],
    network.links.at[link_2_name, "p_nom_opt"],
)
assert_array_almost_equal(
    network_r.storage_units.at[su_to_replace, "p_nom_opt"],
    network.links.at[link_1_name, "p_nom_opt"]
    * network_r.storage_units.at[su_to_replace, "efficiency_dispatch"],
)

"""""
Übernommen von Wind_power Skript 
"""""

fn = "https://raw.githubusercontent.com/PyPSA/powerplantmatching/master/powerplants.csv"
ppl = pd.read_csv(fn, index_col=0)

ppl.plot.scatter("lon", "lat", s=ppl.Capacity / 1e3)

#geometry = gpd.points_from_xy(ppl["lon"], ppl["lat"])
# gpd.points_from_xy wandelt die Koordinaten in einen Punkt um (über shapely.point)
gdf = gpd.GeoDataFrame(ppl, geometry=geometry, crs=4326)

gdf.plot(
    column="Fueltype",
    markersize=gdf.Capacity / 1e2,
)

# df.explore(column="Fueltype")

ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()

fig = plt.figure(figsize=(7, 7))

# ax = plt.axes(projection=ccrs.PlateCarree())

gdf.plot(
    ax=ax,
    column="Fueltype",
    markersize=gdf.Capacity / 1e2,
)

# ax.add_feature(cartopy.feature.BORDERS, color="grey", linewidth=0.5)
# ax.add_feature(cartopy.feature.OCEAN, color="azure")
# ax.add_feature(cartopy.feature.LAND, color="cornsilk")
# ax.set_extent([5, 16, 47, 55])

crs = ccrs.AlbersEqualArea()

ax = plt.axes(projection=crs)

gdf.to_crs(crs.proj4_init).plot(
    ax=ax,
    column="Fueltype",
    markersize=gdf.Capacity / 1e2,
)

ax.coastlines()

fig.show()