#%%
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
current_dir = os.getcwd()
data_path = os.path.join(current_dir,'../../data/W2/')
data_path = data_path.replace('\\', '/')
if not os.path.exists(data_path):
   os.makedirs(data_path)
# %%
# Load data
df_v2 = pd.read_csv(data_path + 'athletes.csv')
# %%
df_v2.head(3)
# %%
sns.set_style('darkgrid')
sns.histplot(data = df_v2, x = 'region', hue = 'gender')
plt.title('Region')
plt.xlabel('Region')
plt.xticks(rotation=90)
plt.show()
# %%
sns.histplot(data = df_v2, x ='BMI', hue = 'gender', kde = True, bins = 30)
plt.title('BMI')
plt.xlabel('BMI')
plt.show()
# %%
sns.boxplot(data=df_v2, x ='total_lift', y = 'region', hue = df_v2['gender'])
plt.title('Total Lift')
plt.xlabel('Region')
plt.xticks(rotation=0)
plt.show()
# %%
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(10, 6), nrows=2, ncols=2)
xlabel = 'normalised lift by body weight'
sns.histplot(data = df_v2, x = 'deadlift_norm', hue = 'gender', ax=ax[0,0], kde = True)
ax[0,0].set_title('Deadlift')
ax[0,0].set_xlabel(xlabel)
sns.histplot(data = df_v2, x = 'candj_norm', hue = 'gender',ax=ax[0,1], kde = True)
ax[0,1].set_title('Clean and Jerk')
ax[0,1].set_xlabel(xlabel)
sns.histplot(data = df_v2, x = 'snatch_norm', hue = 'gender',ax=ax[1,0], kde = True)
ax[1,0].set_title('Snatch')
ax[1,0].set_xlabel(xlabel)
sns.histplot(data = df_v2, x = 'backsq_norm', hue = 'gender',ax=ax[1,1],kde = True)
ax[1,1].set_title('Back Squat')
ax[1,1].set_xlabel(xlabel)
plt.tight_layout()
plt.show()
# %%
sns.histplot(data = df_v2, x = 'total_lift', hue = 'gender',kde = True)
plt.title('Deadlift')
plt.xlabel(xlabel)
plt.show()
# %%
fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=3, tight_layout=True)
fig.suptitle('Athlete Lifestyle', fontsize=16)
ax[0,0].bar('Rec. Sports',df_v2['rec_sports'].sum())
ax[0,0].bar('HS Sports',df_v2['high_sports'].sum())
ax[0,0].bar('College Sports',df_v2['col_sports'].sum())
ax[0,0].bar('Prof. Sports',df_v2['pro_sports'].sum())
ax[0,0].bar('No Background', df_v2['no_bg'].sum())
ax[0,0].set_ylabel('Count')
ax[0,0].set_xticklabels(['Rec. Sports','HS Sports','College Sports','Prof. Sports','No Background'], rotation=90)
ax[0,0].set_title('Sports Background')

ax[0,1].bar('Exp. Coach',df_v2['exp_coach'].sum())
ax[0,1].bar('Exp. Alone',df_v2['exp_alone'].sum())
ax[0,1].bar('Exp. Courses',df_v2['exp_courses'].sum())
ax[0,1].bar('Exp. Crossfit Trainer',df_v2['exp_trainer'].sum())
ax[0,1].bar('Exp. Lvl 1', df_v2['exp_level1'].sum())
ax[0,1].set_ylabel('Count')
ax[0,1].set_xticklabels(['Exp. Coach','Exp. Alone','Exp. Courses','Exp. Crossfit Trainer','Exp. Lvl 1'], rotation=90)
ax[0,1].set_title('Experience')

ax[0,2].bar('Rest 4+',df_v2['rest_plus'].sum())
ax[0,2].bar('Rest <4',df_v2['rest_minus'].sum())
ax[0,2].set_ylabel('Count')
ax[0,2].set_xticklabels(['Rest 4+','Rest <4'], rotation=90)
ax[0,2].set_title('Rest')

ax[1,0].bar('1 workout',df_v2['sched_0extra'].sum())
ax[1,0].bar('>1 workout 1 day',df_v2['sched_1extra'].sum())
ax[1,0].bar('>1 workout 2 days',df_v2['sched_2extra'].sum())
ax[1,0].bar('>1 workout 3+ days',df_v2['sched_3extra'].sum())
ax[1,0].set_ylabel('Count')
ax[1,0].set_xticklabels(['1 workout','>1 workout 1 day','>1 workout 2 days','>1 workout 3+ days'], rotation=90)
ax[1,0].set_title('Schedule')

ax[1,1].bar('<0.5 yrs',df_v2['exp_lt6mo'].sum())
ax[1,1].bar('0.5-1 yrs',df_v2['exp_6to12mo'].sum())
ax[1,1].bar('1-2 yrs',df_v2['exp_1to2yrs'].sum())
ax[1,1].bar('2-4 yrs',df_v2['exp_2to4yrs'].sum())
ax[1,1].bar('4+ yrs',df_v2['exp_4plus'].sum())
ax[1,1].set_ylabel('Count')
ax[1,1].set_xticklabels(['<0.5 yrs','0.5-1 yrs','1-2 yrs','2-4 yrs','4+ yrs'], rotation=90)
ax[1,1].set_title('Crossfit Experience')

ax[1,2].bar('Eats \nconvinient',df_v2['eat_conv'].sum())
ax[1,2].bar('Cheat meals',df_v2['eat_cheat'].sum())
ax[1,2].bar('Eats \nQuality',df_v2['eat_quality'].sum())
ax[1,2].bar('Eats \nPaleo',df_v2['eat_paleo'].sum())
ax[1,2].bar('Measures wt',df_v2['eat_weigh'].sum())
ax[1,2].set_ylabel('Count')
ax[1,2].set_xticklabels(['Eats \nconvinient','Cheat meals','Eats \nQuality','Eats \nPaleo','Measures wt'], rotation=90)
ax[1,2].set_title('Diet')
# %%
num_cols = []
for i in df_v2.columns:
    if df_v2[i].dtype == 'int64':
        num_cols.append(i)
df_corr = df_v2[num_cols].corr()
plt.figure(figsize=(15,10))
plt.style.use('ggplot')
sns.heatmap(df_corr, annot=False, cmap='coolwarm')
plt.grid(True)
plt.show()
# %%
model_features = ['age','gender','BMI','rec_sports','high_sports','col_sports',\
                  'pro_sports','no_bg','exp_coach','exp_alone','exp_courses',\
                   'life_changing','exp_trainer','exp_level1','exp_start_nr',\
                    'rest_plus','rest_minus','rest_sched', 'sched_0extra', 'sched_1extra',\
                    'sched_2extra', 'sched_3extra', 'sched_nr', 'rest_nr', 'exp_1to2yrs',\
                    'exp_2to4yrs', 'exp_4plus', 'exp_6to12mo', 'exp_lt6mo', 'eat_conv',\
                    'eat_cheat', 'eat_quality', 'eat_paleo', 'eat_weigh','total_lift']
# %%
target = 'total_lift'
features = df_v2[model_features]
X = df_v2[model_features]
y = df_v2[target]
# %%
from sklearn.model_selection import train_test_split,cross_val_score
import xgboost as xgb
# %%
df_v2
# %%
