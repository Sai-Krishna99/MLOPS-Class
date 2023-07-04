# %%
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer,DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy_statement
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
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
# Drop unnecessary columns.
drop_cols = ['eat', 'background', 'experience', 'schedule', 'howlong', 'height', 'weight', 'candj', 'snatch', 'deadlift',
       'backsq', 'candj_norm', 'snatch_norm', 'backsq_norm', 'deadlift_norm']
df_v2.drop(drop_cols, axis=1, inplace=True)
# %%
# One hot encode the region column.
df_v2 = pd.get_dummies(df_v2, columns=['region'], dtype=int)
# %%
#train test split
X = df_v2.drop(['total_lift'], axis=1)
y = df_v2['total_lift']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
# Convert the data to TensorFlow tensors
X_train_tf = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train.values.reshape(-1, 1), dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
# %%
# Define the differentially private optimizer
dp_optimizer = DPAdamGaussianOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=1
)
# %%
# Define the GLM model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear')
])
# %%
# Compile the model with the differentially private optimizer
model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=dp_optimizer)
# %%
# Train the model
model.fit(X_train_tf, y_train_tf, epochs=10, batch_size=32)
# %%
#Define the privacy parameters
epsilon = 1.0
delta = 1e-5
l2_norm_clip = 1.0
noise_multiplier = 1.1
# %%
#Compute the DP (privacy guarantee epsilon)
num_examples = len(X_train)
batch_size = 64
epochs = num_examples // batch_size
## https://github.com/tensorflow/privacy/blob/v0.8.10/tensorflow_privacy/privacy/analysis/compute_dp_sgd_privacy.py
privacy_guarantee = compute_dp_sgd_privacy_statement(number_of_examples=num_examples,
                                              batch_size=batch_size,
                                              noise_multiplier=noise_multiplier,
                                              num_epochs=epochs,
                                              delta=1e-5)

print("Privacy Guarantee Epsilon:", privacy_guarantee)