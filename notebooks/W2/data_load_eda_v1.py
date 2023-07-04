#%%
# Load the dependencies and set the path for the data
import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from google.cloud import storage
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
current_dir = os.getcwd()
data_path = os.path.join(current_dir,'../../data/W2/')
data_path = data_path.replace('\\', '/')
if not os.path.exists(data_path):
   os.makedirs(data_path)
#%%
#Download the data from GCP or Google Drive
if input("Do you want to download the data from GCP? (y/n)") == 'y':
    # Download data from GCP
    #bucket key would be in the .json file added to the github secrets
    client = storage.Client.from_service_account_json('../savvy-pad-385820-cf9cf5d85030.json')
    bucket = client.get_bucket('sai-nlp-bucket')
    blob = bucket.blob('athletes.csv.zip')
    blob.download_to_filename(data_path + 'athletes.csv.zip')
else:
    # Alternatively, download the data from Google Drive
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    file_id = '1FNfZPJInd6kdbtlD8Gpb2VyKSyOsc6bL'
    file_name = os.path.join(data_path, 'athletes.csv.zip')
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(file_name)
# %%
#Extract the csv from the zipfile and rename the dataset to v1
with zipfile.ZipFile(data_path +'athletes.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(data_path)

original_file_name = data_path + 'athletes.csv'
v1_file_name = data_path + 'athletes.csv'
os.rename(original_file_name, v1_file_name)
# %%
#Load the dataset
df_v1 = pd.read_csv(data_path+ 'athletes.csv')
df_v1.head(3)
# %%
df_v1.info()
# %%
nulls_df = pd.DataFrame(df_v1.isnull().sum().sort_values(ascending=False)/len(df_v1)  * 100, columns=['percentNulls'])
nulls_df
# %%
sns.barplot(x=nulls_df.index, y=nulls_df['percentNulls'])
plt.xticks(rotation=90)
plt.show()
# %%
df_v1['total_lift'] = df_v1['backsq'] + df_v1['candj'] + df_v1['deadlift'] + df_v1['snatch']
df_v1['total_lift'].describe()
# %%
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
xlabel = 'Total lift by body weight'
sns.histplot(data = df_v1.dropna(), x = 'deadlift', hue = 'gender', ax=ax[0,0], kde = True)
ax[0,0].set_title('Deadlift')
ax[0,0].set_xlabel(xlabel)
sns.histplot(data = df_v1.dropna(), x = 'candj', hue = 'gender',ax=ax[0,1], kde = True)
ax[0,1].set_title('Clean and Jerk')
ax[0,1].set_xlabel(xlabel)
sns.histplot(data = df_v1.dropna(), x = 'snatch', hue = 'gender',ax=ax[1,0], kde = True)
ax[1,0].set_title('Snatch')
ax[1,0].set_xlabel(xlabel)
sns.histplot(data = df_v1.dropna(), x = 'backsq', hue = 'gender',ax=ax[1,1],kde = True)
ax[1,1].set_title('Back Squat')
ax[1,1].set_xlabel(xlabel)
plt.tight_layout()
plt.show()
# %%
sns.histplot(data = df_v1.dropna(), x = 'total_lift', hue = 'gender',kde = True)
plt.title('Deadlift')
plt.xlabel(xlabel)
plt.show()
# %%
df_v1.columns
# %%
sns.histplot(data = df_v1.dropna(), x = 'region', hue = 'gender')
plt.title('Region')
plt.xlabel('Region')
plt.xticks(rotation=90)
plt.show()
# %%
