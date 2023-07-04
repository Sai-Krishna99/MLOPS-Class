# %%
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,explained_variance_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
current_dir = os.getcwd()
data_path = os.path.join(current_dir,'../../data/W2/')
data_path = data_path.replace('\\', '/')
if not os.path.exists(data_path):
   os.makedirs(data_path)
# %%
# Load data
df_v1 = pd.read_csv(data_path + 'athletes.csv')
df_v1.head(2)
#%%
df_v1.columns
# %%
# calculate total lift
df_v1['total_lift'] = df_v1['backsq'] + df_v1['candj'] + df_v1['deadlift'] + df_v1['snatch']
# %%
drop_cols = ['athlete_id','name','affiliate','helen','grace','filthy50',\
             'fgonebad','run400','run5k','candj','backsq','deadlift','snatch',\
             'pullups','region','team','weight','height','fran','fgonebad',\
             'team']
df_v1.drop(drop_cols, axis=1, inplace=True)
df_v1.shape
# %%
df_v1.dropna(inplace=True)
df_v1.shape
# %%
cat_features = df_v1.select_dtypes(include=['object']).columns
cat_features
# %%
# label encoding
le = LabelEncoder()
label_mappings = {}
for col in cat_features:
    df_v1[col] = le.fit_transform(df_v1[col])
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
df_v1.head(2)
# %%
# Print label mappings
for col, mappings in label_mappings.items():
    print(f"Column: {col}")
    for label, encoded_value in mappings.items():
        print(f"{label}: {encoded_value}")
    print()
# %%
#train test split
X = df_v1.drop(['total_lift'], axis=1)
y = df_v1['total_lift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
# Baseline model Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# %%
y_pred = rf_reg.predict(X_test)

explained_variance = explained_variance_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Coefficient of Determination (R-squared):", r2)
print("Explained Variance Score:", explained_variance)
# %%
#XGBoost Regressor
xgb_reg = XGBRegressor(random_state=42)
xgb_reg.fit(X_train, y_train)

# %%
y_pred = xgb_reg.predict(X_test)
explained_variance = explained_variance_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Coefficient of Determination (R-squared):", r2)
print("Explained Variance Score:", explained_variance)
# %%
#The baseline models with no cleaning or feature extraction performs worse than a simple horizontal line.
#the model is not able to capture the variance in the data