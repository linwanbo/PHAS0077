
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
diabet = pd.read_csv("diabetic_data.csv")
diabet.readmitted.value_counts()
import seaborn as sns
sns.set()

diabet = diabet.replace("?",np.nan)

diabet = diabet.replace({"NO":0,
                         "<30":1,
                         ">30":0})

print(diabet.readmitted.value_counts())


# explore the distribution of readmitted
sns.countplot(x = "readmitted", data = diabet)
plt.title("Distribution of Target Values")
plt.show()

# Pie chart
diabet.readmitted.value_counts().plot.pie(autopct = "%.1f%%")
plt.title("Proportion of Target Value")
plt.show()


diabet.race.value_counts().plot.pie(autopct = "%1000.1f%%")
plt.title("proportion of Race values")
plt.show()


# explore the distribution of race
mapped_race = {"Asian":"Other","Hispanic":"Other"}
diabet.race = diabet.race.replace(mapped_race)

sns.countplot(x="race", data = diabet)
plt.title("Number of Race values")
plt.show()

print("Proportion of Race After the Mapping")
print(diabet.race.value_counts(normalize= True)*100)



sns.countplot(x="race", hue= "readmitted", data = diabet)
plt.title("Readmitted - Race")
plt.show()

sns.catplot(x = "race", y = "readmitted",
            data = diabet, kind = "bar", height= 5)
plt.title("Readmitted Probability")
plt.show()

# explore the distribution of gender
diabet = diabet.drop(diabet.loc[diabet["gender"]=="Unknown/Invalid"].index, axis=0)


sns.countplot(x = "gender", hue = "readmitted", data = diabet)
plt.title("Gender - Readmitted")
plt.show()

g = sns.catplot(x = "gender",y = "readmitted",
                data = diabet, kind = "bar", height= 5)
g.set_ylabels("Readmitted Probability")
plt.show()

# explore the distribution of age
sns.countplot(x="age", data = diabet)
plt.xticks(rotation = 90)
plt.show()


g = sns.catplot(x = "age", y = "readmitted", data = diabet,
                   kind = "bar", height = 5)
g.set_ylabels("Readmitted Probability")
plt.show()

# explore the distribution of Time in Hospital
sns.countplot(x="time_in_hospital", data = diabet,
              order = diabet.time_in_hospital.value_counts().index)
plt.show()

print(diabet.time_in_hospital.value_counts())

fig = plt.figure(figsize=(10,5))

readmitted = 0
ax = sns.kdeplot(diabet.loc[(diabet.readmitted == 0), "time_in_hospital"],
                 color = "b", shade = True, label = "Not Readmitted")

ax = sns.kdeplot(diabet.loc[(diabet.readmitted == 1), "time_in_hospital"],
                 color = "r", shade = True, label = "Readmitted")
ax.legend(loc="upper right")

ax.set_xlabel("Time in Hospital")
ax.set_ylabel("Frequency")
ax.set_title("Time in Hospital - Readmission")
plt.show()



# explore the distribution of dig
def map_diagnosis(data, cols):
    for col in cols:
        data.loc[(data[col].str.contains("V")) | (data[col].str.contains("E")), col] = -1
        data[col] = data[col].astype(np.float16)

    for col in cols:
        data["temp_diag"] = np.nan
        data.loc[(data[col]>=390) & (data[col]<=459) | (data[col]==785), "temp_diag"] = "Circulatory"
        data.loc[(data[col]>=460) & (data[col]<=519) | (data[col]==786), "temp_diag"] = "Respiratory"
        data.loc[(data[col]>=520) & (data[col]<=579) | (data[col]==787), "temp_diag"] = "Digestive"
        data.loc[(data[col]>=250) & (data[col]<251), "temp_diag"] = "Diabetes"
        data.loc[(data[col]>=800) & (data[col]<=999), "temp_diag"] = "Injury"
        data.loc[(data[col]>=710) & (data[col]<=739), "temp_diag"] = "Muscoloskeletal"
        data.loc[(data[col]>=580) & (data[col]<=629) | (data[col] == 788), "temp_diag"] = "Genitourinary"
        data.loc[(data[col]>=140) & (data[col]<=239), "temp_diag"] = "Neoplasms"

        data["temp_diag"] = data["temp_diag"].fillna("Other")
        data[col] = data["temp_diag"]
        data = data.drop("temp_diag", axis=1)

    return data
diabet = map_diagnosis(diabet,["diag_1","diag_2","diag_3"])
def plot_diags(col,data):
    sns.countplot(x = col, data = data,
            order = data[f"{col}"].value_counts().index)
    plt.xticks(rotation = 20,fontsize=8)
    plt.title(col)
    plt.show()

diag_cols = ["diag_1","diag_2","diag_3"]

for diag in diag_cols:
    plot_diags(diag,diabet)



# explore the distribution of drug
drug_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
        'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone', 'tolbutamide']

for col in drug_cols:
    diabet[col] = diabet[col].replace(['No','Steady','Up','Down'],[0,1,1,1])
    diabet[col] = diabet[col].astype(int)

def explore_drug(drugs):
    i=0
    for drug in drugs:
        g=sns.countplot(x=drug,
                      hue="readmitted",
                      data=diabet,
                      order=[1])
        i = i + 1
        if i!=22:

            plt.subplot(4,5,i)
            g.get_legend().remove()
            g.xaxis.set_visible(False)
            g.yaxis.set_visible(False)
            g.set(xticklabels=[])
            g.set(xlabel=None)
            g.tick_params(bottom=False)


    plt.show()



explore_drug(drug_cols)



# explore the distribution of A1ctest
diabet["A1Cresult"] = diabet["A1Cresult"].replace({">7":2,
                                           ">8":2,
                                           "Norm":1,
                                           "None":0})

sns.countplot(x = "A1Cresult", data = diabet)
plt.show()

sns.countplot(x = "A1Cresult",hue = "readmitted", data = diabet)
plt.show()

print(diabet.A1Cresult.value_counts())