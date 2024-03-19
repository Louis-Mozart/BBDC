import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

data = pd.read_csv("/home/leo/sciebo/BBDC/bbdcdata_10000.csv")
print(list(data.columns))
numeric_features = ['timestamp', 'ppg_filled', 
                    'hr_filled', 'hrIbi_filled', 'x_filled', 'y_filled', 
                    'z_filled']
categorical_features = ['sessionId', 'affect', 'hr_status_filled', 
                        'age', 'gender', 'fairNumber']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_filtered = data[data['context'].notna()]

y = data_filtered['context']
y = pd.DataFrame(y)
X = data_filtered.drop('context', axis=1)

x_processed = preprocessor.fit_transform(X)

onehot = OneHotEncoder(sparse_output=False)
y_processed = onehot.fit_transform(y)
y_cat = y.astype('category')
y_uint =  y_cat["context"].cat.codes.astype('uint8')

test_size = 10  # Number of test samples
validation_size = 0.2  # Proportion of validation set size relative to (training + validation)

test_proportion = test_size / len(x_processed)
x_temp, x_test, y_temp, y_test = train_test_split(x_processed, y_uint, test_size=test_proportion, random_state=42)

x_train, x_validate, y_train, y_validate = train_test_split(x_temp, y_temp, test_size=validation_size, random_state=42)

x_train_3D = np.expand_dims(x_train, axis=1) 
x_validate_3D = np.expand_dims(x_validate, axis=1) 
x_test_3D = np.expand_dims(x_test, axis=1) 


model = keras.Sequential()
model.add(layers.GRU(1000, input_shape=(1,68)))
keras.layers.ConvLSTM2D(500, 5)
keras.layers.SimpleRNN(20)
model.add(layers.BatchNormalization())
model.add(layers.Dense(4))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	

model.fit(
    x_train_3D, y_train, validation_data=(x_validate_3D, y_validate), batch_size=64, epochs=10
)


#############################

model2 = XGBClassifier(objective='multi:softprob')


model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
table = [list(y_test),list(y_pred)]
for i in range(10):
    print(table[0][i], table[1][i])


###########################

#Work in progress

import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
import os
import tensorflow as tf
import warnings
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from xgboost import plot_importance

xgb_model = xgb.XGBRegressor(gamma=1)
xgb_model.fit(y_train, y_train)

pred_validate = xgb_model.predict(x_validate)

mae = mean_absolute_error(y_validate, pred_validate)

pred_test = xgb_model.predict(x_test)

xgb_model.predict(x_validate)
mae = mean_absolute_error(y_validate, pred_validate)
mae_xgboost = mae
