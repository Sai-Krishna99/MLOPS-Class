# %%
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
df_v2 = pd.read_csv(data_path + 'athletes.csv')
df_v2.head(2)
# %%
df_v2.columns
# %%
cat_features = df_v2.select_dtypes(include=['object']).columns
cat_features
# %%
drop_cols = ['eat', 'background', 'experience', 'schedule', 'howlong', 'height', 'weight', 'candj', 'snatch', 'deadlift',
       'backsq', 'candj_norm', 'snatch_norm', 'backsq_norm', 'deadlift_norm']
df_v2.drop(drop_cols, axis=1, inplace=True)
# %%
#One hot encode the region column
df_v2 = pd.get_dummies(df_v2, columns=['region'], dtype=int)
df_v2.head(2)
# %%
#train test split
X = df_v2.drop(['total_lift'], axis=1)
y = df_v2['total_lift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
# Random Forest Regressor
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
feature_importances = rf_reg.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.head(10)
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
feature_importances = xgb_reg.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.head(10)
# %%
# GLM
glm_model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Gaussian())
glm_results = glm_model.fit()
# %%
y_pred = glm_results.predict(sm.add_constant(X_test))

#Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

#Print the performance metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Coefficient of Determination (R-squared):", r2)
print("Explained Variance Score:", explained_variance)
# %%
