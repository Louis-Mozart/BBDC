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

#### Load data

data = pd.read_csv("/home/leo/sciebo/BBDC/training_set_context.csv")
test_set = pd.read_csv("/home/leo/sciebo/BBDC/test_set_context.csv")
print(data.columns.values)
data = data.drop('combined', axis=1)
test_set = test_set.drop('combined', axis=1)
data = data.drop('timestamp', axis=1)
test_set = test_set.drop('timestamp', axis=1)
data = data.drop('age', axis=1)
test_set = test_set.drop('age', axis=1)
data = data.drop('gender', axis=1)
test_set = test_set.drop('gender', axis=1)

data_filtered = data[data['context'].notna()]

y = data_filtered['context']
y = pd.DataFrame(y)
X = data_filtered.drop('context', axis=1)

y_test = test_set['context']
y_test = pd.DataFrame(y_test)
x_test = test_set.drop('context', axis=1)

print(X.columns.values)

numeric_features = [#'timestamp', 
                    'ppg_filled', 
                    'hr_filled', 'hrIbi_filled', 'x_filled', 'y_filled', 
                    'z_filled']
categorical_features = ['sessionId', 'affect', 'hr_status_filled',
                        #'age', 'gender', 
                        'fairNumber'
                        ]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')


x_test['source'] = 'x_test'
X['source'] = 'X'

# Concatenate
combined_Xes = pd.concat([X, x_test], ignore_index=True)

x_combined_processed = preprocessor.fit_transform(combined_Xes)
source_column = x_combined_processed[:, 63]
x_processed = x_combined_processed[source_column == "X"]
x_test_processed = x_combined_processed[source_column == "x_test"]
x_processed = np.delete(x_processed, 63, axis=1)
x_test_processed = np.delete(x_test_processed, 63, axis=1)
x_processed = x_processed.astype(np.float64)
x_test_processed = x_test_processed.astype(np.float64)

onehot = OneHotEncoder(sparse_output=False)
y_cat = y.astype('category')
y_uint =  y_cat["context"].cat.codes.astype('uint8')
y_test_cat = y_test.astype('category')
y_test_uint =  y_test_cat["context"].cat.codes.astype('uint8')

validation_size = 0.25  # Proportion of validation set size relative to (training + validation)

#
x_train, x_validate, y_train, y_validate = train_test_split(x_processed, y_uint, test_size=validation_size, random_state=42)

x_train_3D = np.expand_dims(x_train, axis=1) 
x_validate_3D = np.expand_dims(x_validate, axis=1) 
x_test_3D = np.expand_dims(x_test_processed, axis=1) 

################ keras
#%%

np.random.seed(1888)

tf.random.set_seed(1888)

model = keras.Sequential()
model.add(layers.GRU(1000, input_shape=(1,63)))
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

results = model.evaluate(x_test_3D, y_test_uint)
