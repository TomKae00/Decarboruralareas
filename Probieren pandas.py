import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['row1', 'row2'])
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]}, index=['row2', 'row3'])

df2_aligned = df2.reindex(df1.index, fill_value=0)

result = df1 + df2_aligned

print(result)

#titanic = pd.read_csv("data/titanic.csv")
#ages = titanic["Age"]
#print(ages.head(10))
#print(type(titanic["Age"]), titanic["Age"].shape)
#age_sex = titanic[["Age", "Sex"]]
#print(age_sex.head(10))
#print(type(titanic[["Age", "Sex"]]), titanic[["Age", "Sex"]].shape)
#above_35 = titanic[titanic["Age"] > 35]
#print(above_35.head(10))
#print(titanic["Age"] > 35)
#print(above_35.shape)
#class_23 = titanic[titanic["Pclass"].isin([2, 3])]
#print(class_23.head(10))
#class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
#print(class_23.head(10))
#age_no_na = titanic[titanic["Age"].notna()]
#print(age_no_na.head(10))
#print(age_no_na.shape)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# select specific rows and columns from a DataFrame
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#adult_names = titanic.loc[titanic["Age"] > 35, "Name"]
#print(adult_names)
#print(titanic.iloc[9:25, 2:5])
#titanic.iloc[0:3, 3] = "anonymous"




#costs = pd.read_csv("data/costs_2020.csv")
#print(costs)
#print(costs.iloc[159, 2])

#titanic = pd.read_csv("data/titanic.csv")
#print(titanic.head(8))
#print(titanic.dtypes)
#titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False)
# Optional: Anzeigen einer Bestätigungsmeldung
#print("DataFrame erfolgreich in Excel-Datei gespeichert.")

#df = pd.DataFrame(
#        {
#            "Name": [
#                "Braund, Mr. Owen Harris",
#                "Allen, Mr. William Henry",
#               "Bonnell, Miss. Elizabeth",
#           ],
#           "Age": [22, 35, 58],
#           "Sex": ["male", "male", "female"],
#        }
#   )
#print(df["Age"])