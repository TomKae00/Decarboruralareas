import pandas as pd

# Erstellen Sie Zeitstempel für stündliche Intervalle für einen Monat
timestamps = pd.date_range(start="2023-01-01", end="2023-01-31", freq="H")

# Wandeln Sie die Zeitstempel in ein DataFrame um
df = pd.DataFrame({"timestamp": timestamps})

# Exportieren Sie das DataFrame in eine CSV-Datei
df.to_csv('timestamps.csv', index=False)