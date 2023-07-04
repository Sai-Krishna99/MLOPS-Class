#%%
# Load the dependencies and set the path for the data
import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
current_dir = os.getcwd()
data_path = os.path.join(current_dir,'../../data/W2/')
data_path = data_path.replace('\\', '/')
if not os.path.exists(data_path):
   os.makedirs(data_path)
# %%
#Load the dataset
df_v1 = pd.read_csv(data_path+ 'athletes.csv')
df_v1.head(3)
# %%
#Check the shape and info of the dataset
print(f"The shape of the dataset is: {df_v1.shape}")
df_v1.info()
# %%
#Check the null values in the dataset
plt.style.use('ggplot')
plt.figure(figsize = (8,5))
df_nulls = pd.DataFrame(df_v1.isnull().sum()/df_v1.shape[0] * 100, columns = ['Nulls%'])
sns.barplot(x = df_nulls.index, y = df_nulls['Nulls%'])
plt.xticks(rotation = 90, fontsize = 8)
plt.title('Nulls Percentage in the Dataset')
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(df_v1.isnull(), yticklabels=False)
plt.title('Null Values Heat Map for the Dataset')
plt.show()
# %%
#Drop the columns with more null values/not needed ones and drop the nulls in the remaining columns
df_v1 = df_v1.drop(columns = ['athlete_id','name','team','affiliate','fran','helen','grace',\
    'filthy50','fgonebad','run400','run5k','pullups','train'])

df_v1.dropna(subset=['region','gender','age','height','weight','candj','snatch',\
    'deadlift','backsq','eat','background','experience','schedule','howlong'], inplace=True)
# %%
#Check the nulls again
plt.figure(figsize=(10,5))
sns.heatmap(df_v1.isnull(), yticklabels=False)
plt.title('Null Values Heat Map for the Dataset')
plt.show()
# %%
#Now check for the shape and info of the dataset
print(f"The shape of the dataset is: {df_v1.shape}")
df_v1.info()
# %%
#Check and remove the outliers in the dataset
df_v1 = df_v1[(df_v1['age'] > 18) & (df_v1['age'] < 80)] #Only consider the athletes between 18 and 80 years of age
df_v1 = df_v1[(df_v1['height'] > 48) & (df_v1['height'] < 84)] #Only consider the athletes between 4 and 7 feet tall
df_v1 = df_v1[df_v1['weight'] < 1000] #Only consider the athletes less than 500 pounds
df_v1 = df_v1[(df_v1['gender']!= '--')] #include only male and female
df_v1 = df_v1[(df_v1['snatch'] > 0) & (df_v1['snatch'] < 500)] #Only consider the athletes less than 500 pounds
df_v1 = df_v1[(df_v1['candj'] > 0) & (df_v1['candj'] < 500)] #Only consider the athletes less than 500 pounds
df_v1 = df_v1[(df_v1['backsq'] > 0) & (df_v1['backsq'] < 1000)] #Only consider the athletes less than 500 pounds
df_v1 = df_v1[(df_v1['deadlift'] > 0) & (df_v1['deadlift'] < 1000)] #Only consider the athletes less than 500 pounds
## Remove hihgly under weight and over weight people
df_v1['BMI'] = df_v1['weight']*0.453592/np.square(df_v1['height']*0.0254)
df_v1 = df_v1[(df_v1['BMI']>=17)&(df_v1['BMI']<=50)]
#%%
df_v1.describe()
# %%
df_v1.columns
# %%
#Remove the records with values as decline to answer
decline_cols = ['eat','background','experience','schedule','howlong']
df_v1 = df_v1.replace('Decline to answer|', np.nan)
df_v1.dropna(subset=decline_cols, inplace=True)
df_v1.shape
# %%
## Feature engineering and extraction
#Background
df_v1['rec_sports'] = df_v1['background'].apply(lambda x: 1 if 'I regularly play recreational sports' in x else 0)
df_v1['high_sports'] = df_v1['background'].apply(lambda x: 1 if 'I played youth or high school level sports' in x else 0)
df_v1['col_sports'] = df_v1['background'].apply(lambda x: 1 if 'I played college sports' in x else 0)
df_v1['pro_sports'] = df_v1['background'].apply(lambda x: 1 if 'I played professional sports' in x else 0)
df_v1['no_bg'] = df_v1['background'].apply(lambda x: 1 if 'I have no athletic background besides CrossFit' in x else 0)
# remove all others which show not null and no background
df_v1 = df_v1[((df_v1['rec_sports'] == 1) | (df_v1['col_sports'] == 1) |
                (df_v1['high_sports'] == 1) | (df_v1['pro_sports'] == 1) |
                (df_v1['no_bg'] == 1))]
df_v1.shape
# %%
#Experience
df_v1['exp_coach'] = df_v1['experience'].apply(lambda x: 1 if 'I began CrossFit with a coach' in x else 0)
df_v1['exp_alone'] = df_v1['experience'].apply(lambda x: 1 if 'I began CrossFit by trying it alone' in x else 0)
df_v1['exp_courses'] = df_v1['experience'].apply(lambda x: 1 if 'I have attended one or more specialty courses' in x else 0)
df_v1['life_changing'] = df_v1['experience'].apply(lambda x: 1 if 'I have had a life changing experience due to CrossFit' in x else 0)
df_v1['exp_trainer'] = df_v1['experience'].apply(lambda x: 1 if 'I train other people' in x else 0)
df_v1['exp_level1'] = df_v1['experience'].apply(lambda x: 1 if 'I have completed the CrossFit Level 1 certificate course' in x else 0)

# Delete rows with contradictory answers
df_v1 = df_v1[~((df_v1['exp_coach'] == 1) & (df_v1['exp_alone'] == 1))]

# Create a 'no response' option for coaching start
df_v1['exp_start_nr'] = df_v1.apply(lambda row: 1 if row['exp_coach'] == 0 and row['exp_alone'] == 0 else 0, axis=1)
df_v1.shape
# %%
#Schedule
df_v1['rest_plus'] = df_v1['schedule'].apply(lambda x: 1 if 'I typically rest 4 or more days per month' in x else 0)
df_v1['rest_minus'] = df_v1['schedule'].apply(lambda x: 1 if 'I typically rest fewer than 4 days per month' in x else 0)
df_v1['rest_sched'] = df_v1['schedule'].apply(lambda x: 1 if 'I strictly schedule my rest days' in x else 0)

df_v1['sched_0extra'] = df_v1['schedule'].apply(lambda x: 1 if 'I usually only do 1 workout a day' in x else 0)
df_v1['sched_1extra'] = df_v1['schedule'].apply(lambda x: 1 if 'I do multiple workouts in a day 1x a week' in x else 0)
df_v1['sched_2extra'] = df_v1['schedule'].apply(lambda x: 1 if 'I do multiple workouts in a day 2x a week' in x else 0)
df_v1['sched_3extra'] = df_v1['schedule'].apply(lambda x: 1 if 'I do multiple workouts in a day 3+' in x else 0)

# Removing/correcting problematic responses
df_v1 = df_v1[~((df_v1['rest_plus'] == 1) & (df_v1['rest_minus'] == 1))] # You can't have both more than and less than 4 rest days/month

# Points are only assigned for the highest extra workout value (3x only vs. 3x and 2x and 1x if multi selected)
df_v1['sched_0extra'] = np.where((df_v1['sched_3extra'] == 1), 0, df_v1['sched_0extra'])
df_v1['sched_1extra'] = np.where((df_v1['sched_3extra'] == 1), 0, df_v1['sched_1extra'])
df_v1['sched_2extra'] = np.where((df_v1['sched_3extra'] == 1), 0, df_v1['sched_2extra'])
df_v1['sched_0extra'] = np.where((df_v1['sched_2extra'] == 1), 0, df_v1['sched_0extra'])
df_v1['sched_1extra'] = np.where((df_v1['sched_2extra'] == 1), 0, df_v1['sched_1extra'])
df_v1['sched_0extra'] = np.where((df_v1['sched_1extra'] == 1), 0, df_v1['sched_0extra'])

# Adding no response columns
df_v1['sched_nr'] = df_v1.apply(lambda row: 1 if row['sched_0extra'] == 0 and row['sched_1extra'] == 0 and row['sched_2extra'] == 0 and row['sched_3extra'] == 0 else 0, axis=1)
df_v1['rest_nr'] = df_v1.apply(lambda row: 1 if row['rest_plus'] == 0 and row['rest_minus'] == 0 else 0, axis=1)
df_v1.shape
# %%
#Howlong
df_v1['exp_1to2yrs'] = df_v1['howlong'].apply(lambda x: 1 if '1-2 years' in x else 0)
df_v1['exp_2to4yrs'] = df_v1['howlong'].apply(lambda x: 1 if '2-4 years' in x else 0)
df_v1['exp_4plus'] = df_v1['howlong'].apply(lambda x: 1 if '4+' in x else 0)
df_v1['exp_6to12mo'] = df_v1['howlong'].apply(lambda x: 1 if '6-12 months' in x else 0)
df_v1['exp_lt6mo'] = df_v1['howlong'].apply(lambda x: 1 if 'Less than 6 months' in x else 0)

# Keeping only the highest response
df_v1['exp_lt6mo'] = np.where((df_v1['exp_4plus'] == 1), 0, df_v1['exp_lt6mo'])
df_v1['exp_6to12mo'] = np.where((df_v1['exp_4plus'] == 1), 0, df_v1['exp_6to12mo'])
df_v1['exp_1to2yrs'] = np.where((df_v1['exp_4plus'] == 1), 0, df_v1['exp_1to2yrs'])
df_v1['exp_2to4yrs'] = np.where((df_v1['exp_4plus'] == 1), 0, df_v1['exp_2to4yrs'])
df_v1['exp_lt6mo'] = np.where((df_v1['exp_2to4yrs'] == 1), 0, df_v1['exp_lt6mo'])
df_v1['exp_6to12mo'] = np.where((df_v1['exp_2to4yrs'] == 1), 0, df_v1['exp_6to12mo'])
df_v1['exp_1to2yrs'] = np.where((df_v1['exp_2to4yrs'] == 1), 0, df_v1['exp_1to2yrs'])
df_v1['exp_lt6mo'] = np.where((df_v1['exp_1to2yrs'] == 1), 0, df_v1['exp_lt6mo'])
df_v1['exp_6to12mo'] = np.where((df_v1['exp_1to2yrs'] == 1), 0, df_v1['exp_6to12mo'])
df_v1['exp_lt6mo'] = np.where((df_v1['exp_6to12mo'] == 1), 0, df_v1['exp_lt6mo'])
df_v1.shape
# %%
#Eat
df_v1['eat_conv'] = df_v1['eat'].apply(lambda x: 1 if 'I eat whatever is convenient' in x else 0)
df_v1['eat_cheat'] = df_v1['eat'].apply(lambda x: 1 if 'I eat 1-3 full cheat meals per week' in x else 0)
df_v1['eat_quality'] = df_v1['eat'].apply(lambda x: 1 if "I eat quality foods but don't measure the amount" in x else 0)
df_v1['eat_paleo'] = df_v1['eat'].apply(lambda x: 1 if 'I eat strict Paleo' in x else 0)
df_v1['eat_weigh'] = df_v1['eat'].apply(lambda x: 1 if 'I weigh and measure my food' in x else 0)
df_v1.shape
# %%
# Region
regions_usa = ['South East','North East','North Central','Mid Atlantic','South Central',\
               'Central East','South West','Southern California', 'North West','Northern California']
regions_canada = ['Canada East','Canada West']
df_v1['region'] = df_v1['region'].apply(lambda x: 'USA' if x in regions_usa else 'Canada' if x in regions_canada else x)
df_v1.region.value_counts()
# %%
df_v1['gender'] = df_v1['gender'].apply(lambda x: 1 if x=='Male' else 0)
df_v1.gender.value_counts()
# %%
#normalize the deadlift weight to the bodyweight
df_v1['deadlift_norm'] = df_v1['deadlift'] / df_v1['weight']
df_v1['candj_norm'] = df_v1['candj'] / df_v1['weight']
df_v1['snatch_norm'] = df_v1['snatch'] / df_v1['weight']
df_v1['backsq_norm'] = df_v1['backsq'] / df_v1['weight']
#calculate the total lift weight
df_v1['total_lift'] = df_v1['deadlift_norm'] + df_v1['candj_norm'] + df_v1['snatch_norm'] + df_v1['backsq_norm']
df_v1.shape
# %%
#Not tracked anywhere, just for reference
df_v2 = df_v1.copy()
df_v2.to_csv(data_path + 'athletes_v2.csv', index=False)
# %%
#will be used for DVC tracking
df_v2.to_csv(data_path + 'athletes.csv', index=False)