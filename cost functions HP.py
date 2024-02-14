import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load data from CSV file and set the first column as index."""
    return pd.read_csv(file_path, index_col=0)


# Load the data
parameters_tci = load_data('data/table_3_data.csv')
# parameters_cop = load_data('data/table_4_data.csv')

# Constants and Parameters
T_supply_k = 60 + 273.15  # Supply temperature in Kelvin
dT_lift_k = 55  # Temperature lift in Kelvin
COP_carnot = T_supply_k / dT_lift_k
T_source_c = 5  # Source temperature in Celsius


# Functions for Calculations
def calculate_max_supply_temperatures(T_source):
    """Calculate and return the max supply temperatures for various fluids including R717."""
    temps = {'R290': 99.7, 'R1234yf': 95.3, 'R134a': 101.7, 'R600a': 137.3, 'R600': 154.9, 'R245fa': 154.8,
             'R717': 44.6 + 0.9928 * T_source if T_source < 51.20 else 104.20 - 0.1593 * T_source}
    # Convert to Kelvin
    return {fluid: temp + 273.15 for fluid, temp in temps.items()}


max_supply_temps_k = calculate_max_supply_temperatures(T_source_c)


def check_constraints(fluid, T_supply, dT_lift):
    """Check if a fluid meets the supply temperature and dT_lift constraints."""
    return (fluid in max_supply_temps_k and T_supply <= max_supply_temps_k[fluid] and
            dT_lift <= 44) if fluid == 'R717' else True
# Check constraints function


def calculate_tci_strich_1000(D, E, F, T_supply, dT_lift):
    """Calculate TCI for a 1000 kW heat pump."""
    return D + E * dT_lift + F * T_supply


# Calculation Functions
def calculate_scaled_tci(TCI_strich_1000, alpha, X_kW_th):
    """Scale TCI based on heat pump size."""
    return TCI_strich_1000 * (X_kW_th / 1000) ** alpha


#def calculate_pf(A, B, C, COP, COP_strich_1000):
#    """Calculate Power Factor (PF) based on A, B, C, COP, and COP_strich_1000."""
#    return A * (COP / COP_strich_1000) ** B + C


#def calculate_cop_TCImin(a, b, c, T_supply, dT_lift, COP_carnot):
#    """Calculate COP based on parameters a, b, c, T_supply, dT_lift, and COP_carnot."""
#    return (a * T_supply + b * dT_lift + c) * COP_carnot


#def calculate_cop_COPmax(d, e, f, T_supply, dT_lift, COP_carnot):
#    """Calculate COP based on parameters a, b, c, T_supply, dT_lift, and COP_carnot."""
#    return (d * T_supply + e * dT_lift + f) * COP_carnot


def calculate_COP_1000(G, H, I, J, K, T_supply, dT_lift):
    """Calculate COP based on parameters and supply/dT_lift conditions."""
    return G + H * dT_lift + I * T_supply + J * dT_lift**2 + K * T_supply * dT_lift


def calculate_additional_costs(size_mw):
    """Calculate construction, electricity, and heat source investment costs in euros."""
    construction_cost = (0.084311 * size_mw + 0.021769) * 1e6  # Convert from Mio. EUR to EUR
    electricity_cost = (0.12908 * size_mw + 0.01085) * 1e6  # Convert from Mio. EUR to EUR
    heat_source_cost = (0.12738 * size_mw + 5.5007e-6) * 1e6  # Convert from Mio. EUR to EUR
    return construction_cost, electricity_cost, heat_source_cost


"""""
def heat_source_investment_costs(x, source_type):
    # Heat source investment costs (Mio. EUR) differ based on the source type
    if source_type == 'air':
        return 0.12738 * x + 5.5007e-6
    elif source_type == 'excess_heat':
        return 0.091068 * x + 0.10846
    else:
        raise ValueError("Invalid source type")
"""""

# Main Calculation
sizes_kw_th = np.arange(0, 5001, 100)  # Heat pump sizes from 0 to 5000 kW_th
results = {}
for fluid in parameters_tci.columns:
    if not check_constraints(fluid, T_supply_k, dT_lift_k):
        print(f"{fluid} cannot be used due to constraints.")
        continue
    results[fluid] = {'TCI': [], 'COP': [], 'Total Investment Costs': []}
    for size_kw_th in sizes_kw_th:
        size_mw = size_kw_th / 1000  # Convert kW_th to MW
        D, E, F, G, H, I, J, K, alpha = parameters_tci.loc[['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'α'], fluid].values

        TCI_strich_1000 = calculate_tci_strich_1000(D, E, F, T_supply_k, dT_lift_k)
        scaled_tci = calculate_scaled_tci(TCI_strich_1000, alpha, size_kw_th)
        COP_1000 = calculate_COP_1000(G, H, I, J, K, T_supply_k, dT_lift_k)

        construction_cost, electricity_cost, heat_source_cost = calculate_additional_costs(size_mw)
        total_investment_cost = scaled_tci + construction_cost + electricity_cost + heat_source_cost

        results[fluid]['TCI'].append(scaled_tci)
        results[fluid]['COP'].append(COP_1000)
        results[fluid]['Total Investment Costs'].append(total_investment_cost)


df_results = pd.DataFrame.from_dict({(i, j): results[i][j]
                                     for i in results.keys()
                                     for j in results[i].keys()},
                                     orient='index')

# Plotting
plt.figure(figsize=(14, 8))
ax1 = plt.gca()  # Get current axis for plotting Total Investment Costs
ax2 = ax1.twinx()  # Create another axis for COP

for fluid in parameters_tci.columns:
    if fluid in results:
        sizes_MW = sizes_kw_th / 1000  # Convert sizes to MW for plotting
        total_investment_costs = results[fluid]["Total Investment Costs"]
        cop_values = results[fluid]["COP"]

        # Plot Total Investment Costs
        ax1.plot(sizes_MW, total_investment_costs, label=f"Total Investment {fluid}")

        # Plot COP
        ax2.plot(sizes_MW, cop_values, '--', label=f"COP {fluid}")

# Labels, titles, and legends
ax1.set_xlabel('Heat Pump Size (MW)')
ax1.set_ylabel('Total Investment Costs (Mio. EUR)', color='blue')
ax2.set_ylabel('COP', color='green', rotation=270, labelpad=15)
ax1.set_title('Total Investment Costs and COP for Different Heat Pumps')
ax1.legend(loc='upper left', title="Total Investment")
ax2.legend(loc='upper right', title="COP")

ax1.grid(True)
plt.show()
"""""
a) Construction costs (Mio. EUR): f(all)=0.084311*x+0.021769 
b) Electricity investment costs (Mio. EUR): f(all)=0.12908*x+0.01085
c) heat source investment costs (Mio. EUR): 
        f(air)=0.12738*x+5.5007e^-6
        f(excess_heat)=0.091068*x+0.10846
        f(lake)=15% of total invest cost 
        f(river)=15% of total invest cost 
"""""
"""""
costs Water tank: 7450*V^(-0.47) (2015) (Development in Capex depends primarily on the development in steel prices 
-> Formel für V nehmen, dann kann anhand der gewählten VLT und RLT, dT berechnet werden womit dann die Formel
 in abhängigkeit von MWh umgerechnet wird
 
Electric boiler large
Electric boiler small 

Natural Gas DH only:
aus UK Quelle: Werte sind in Pounds, unbedingt in Euro umrechnen (Quelle aus 2023)
size cost
0.5 34000
1   53000
2   79000
5   140000
10  240000
15  320000
20  450000
25  610000

size cost 
H2 Kessel 
0.5 51000
1   67000
2   94000
5   174000
10  285000
15  370000
20  530000
25  767000

Dies sind nur die Kosten für den Kessel Equipment Kosten etc. sollen noch über Danish Energy hinzugefügt werden.
Außerdem Effizienz etc von Danish energy

Wasserstoffspeicher: 
Aufjdenfall diese Wasserstoff packages nutzen 
Kavernen kommen wahrscheinlich nicht in Frage, dafür werden weitere Informationen zu möglichen Standorten benötigt 

Batteriespeicher:
Lithium Ionen ist der geile scheiß 
Ansonsten morgen nochmal mit Mona gucken, was in Frage kommt 
"""""
