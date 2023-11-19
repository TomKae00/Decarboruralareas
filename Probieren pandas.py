import pandas as pd

titanic = pd.read_csv("data/titanic.csv")
ages = titanic["Age"]
print(ages.head(10))
print(type(titanic["Age"]), titanic["Age"].shape)
age_sex = titanic[["Age", "Sex"]]
print(age_sex.head(10))
print(type(titanic[["Age", "Sex"]]), titanic[["Age", "Sex"]].shape)
above_35 = titanic[titanic["Age"] > 35]
print(above_35.head(10))
print(titanic["Age"] > 35)
print(above_35.shape)
class_23 = titanic[titanic["Pclass"].isin([2, 3])]
print(class_23.head(10))
class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
print(class_23.head(10))

costs = pd.read_csv("data/costs_2020.csv")
print(costs)
print(costs.iloc[0, 2])


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