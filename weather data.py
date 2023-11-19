import pandas as pd

# Laden Sie die Heat Demand-CSV-Datei mit Semikolon als Trennzeichen
heat_demand_data = pd.read_csv('heat_demand.csv', parse_dates=['timestamp'],)

# Zeigen Sie die eingelesenen Daten an
print(heat_demand_data)

pd.DataFrame.to_csv(heat_demand_data)

