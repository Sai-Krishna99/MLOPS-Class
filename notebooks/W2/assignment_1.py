# %%
import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from google.cloud import storage
data_path = '../data/'
if not os.path.exists(data_path):
    os.mkdir(data_path)

#%%
# Download data from GCP
client = storage.Client.from_service_account_json('../savvy-pad-385820-cf9cf5d85030.json')
bucket = client.get_bucket('sai-nlp-bucket')
blob = bucket.blob('athletes.csv.zip')
blob.download_to_filename(data_path + 'athletes.csv.zip')
# %%
with zipfile.ZipFile(data_path +'athletes.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(data_path)

#%%
original_file_name = data_path + 'athletes.csv'
v1_file_name = data_path + 'dataset_v1.csv'
os.rename(original_file_name, v1_file_name)
# %%
df_v1 = pd.read_csv(data_path+ 'dataset_v1.csv')
df_v1.head(3)
# %%
print(f"The shape of the dataset is: {df_v1.shape}")
df_v1.info()
# %%
plt.figure(figsize = (8,5))
df_nulls = pd.DataFrame(df_v1.isnull().sum()/df_v1.shape[0] * 100, columns = ['Nulls%'])
sns.barplot(x = df_nulls.index, y = df_nulls['Nulls%'])
plt.xticks(rotation = 90, fontsize = 8)
plt.title('Nulls Percentage in the Dataset')
plt.show()
# %%
plt.figure(figsize=(10,5))
sns.heatmap(df_v1.isnull(), yticklabels=False)
plt.title('Null Values Heat Map for the Dataset')
plt.show()
# %%
## Drop the columns with more null values/not needed ones and drop the nulls in the remaining columns
df_v1 = df_v1.drop(columns = ['athlete_id','name','team','affiliate','fran','helen','grace',\
    'filthy50','fgonebad','run400','run5k','pullups','train'])

df_v1.dropna(subset=['region','gender','age','height','weight','candj','snatch',\
    'deadlift','backsq','eat','background','experience','schedule','howlong'], inplace=True)
# %%
plt.figure(figsize=(10,5))
sns.heatmap(df_v1.isnull(), yticklabels=False)
plt.title('Null Values Heat Map for the Dataset')
plt.show()
# %%
