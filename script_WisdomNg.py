import pandas as pd
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("marketing_campaign.csv", sep="\t")
print("Number of data", len(data))
data.head()

## Check the data and find missing values
data.info()

## Because only income has missing values and it is not alot, we will drop the rows with missing values
data = data.dropna()
data.info()

data['Marital_Status'].astype(str).unique()

# Drop columns with martial status which are yolo and absurd

data = data[data['Marital_Status'] != 'Absurd']
data = data[data['Marital_Status'] != 'YOLO']

data['Marital_Status'].astype(str).unique()

# Create new features to help with the anaylsis
import numpy as np

data['TotalSpending'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

data['Age'] = 2024 - data['Year_Birth']

data['Education'] = data['Education'].map({'Graduation': 'Graduate', 'PhD': 'PostGraduate', 'Master': 'PostGraduate', '2n Cycle': 'UnderGraduate', 'Basic': 'UnderGraduate'})

data['isParent'] = np.where(data['Kidhome'] + data['Teenhome'] > 0, 1, 0)

data["Marital_Status"] = data["Marital_Status"].map({'Single': 'Alone', 'Together': 'NotAlone', 'Married': 'NotAlone', 'Divorced': 'Alone', 'Widow': 'Alone', 'Alone': 'Alone'})

data['FamilySize'] = data["Marital_Status"].map({'Alone': 1, 'NotAlone': 2}) + data["Kidhome"] + data["Teenhome"]

data['Children'] = data["Kidhome"] + data["Teenhome"]

data['CustomerTimeInDays'] = (pd.to_datetime("2024-11-13") - pd.to_datetime(data['Dt_Customer'], dayfirst=True, format='%d-%m-%Y')).dt.days

data['TotalPurchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

data['TotalAcceptedCmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

data = data.drop(columns=['Year_Birth', 'Kidhome', 'Teenhome', 'Dt_Customer', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
                          'NumCatalogPurchases', 'NumStorePurchases', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response'])

data.head()

## Check for columns whch are not unique and drop them
df = data.copy()
df.nunique()

df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])
df.head()

print(df.isnull().sum())

## Turn all the data into numerical

types = (df.dtypes == "object")
columns = list(types[types].index)
encoder = LabelEncoder()
for column in columns:
    df[column] = df[[column]].apply(encoder.fit_transform)

df.head()

sns.heatmap(df.corr().round(2), annot=True)

scaler = StandardScaler()
scaler.fit(df)
scaled_data = pd.DataFrame(scaler.transform(df), columns = df.columns)
scaled_data.head()

pca = PCA(n_components=3)
pca.fit(scaled_data)
pca_df = pd.DataFrame(pca.transform(scaled_data), columns=['PC1', 'PC2', 'PC3'])
pca_df.describe()

elbow = KElbowVisualizer(KMeans(), k=10)
elbow.fit(pca_df)
elbow.show()

kmeans = KMeans(n_clusters=4).fit(scaled_data)
df['Cluster'] = kmeans.fit_predict(scaled_data)
df.head()

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='3d')
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=df['Cluster'])
plt.show()

sns.countplot(x=df["Cluster"])
plt.show()

columns = [ "Education", "Marital_Status", "Income", "Recency", "NumWebVisitsMonth", "Complain", "Age", "isParent", "FamilySize"]

for column in columns:
    plt.scatter(data["TotalSpending"], data[column], c=df["Cluster"])
    plt.ylabel(column)
    plt.xlabel("TotalSpending")
    plt.show()

