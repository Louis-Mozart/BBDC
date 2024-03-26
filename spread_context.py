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
from copy import deepcopy

#### Load data

data = pd.read_csv("/home/leo/sciebo/BBDC/prof_data.csv")
data_1 = data[data['sessionId'] == 1]
# gather test set

test_set_size = 20

# Identify rows where 'context' is not NaN
non_na_indices = data.index[data['context'].notna()].tolist()

# Randomly sample n indices from those identified (without replacement)
# Ensure n does not exceed the number of available non-NaN context rows
random_selected_indices = np.random.choice(non_na_indices, size=min(test_set_size, len(non_na_indices)), replace=False)

# Use the randomly selected indices to get rows from the original DataFrame
test_set = data.loc[random_selected_indices]

data_wo_test = data.drop(random_selected_indices)

## Add lines

data_wo_test.sort_values(by=['sessionId', 'timestamp'], inplace=True)

# Placeholder for updates
updates = {}

# Process each user group
for userId, group in data_wo_test.groupby('sessionId'):
    non_nan_indices = group[group['context'].notna()].index
    for idx in non_nan_indices:
        # Context value to spread
        context_val = group.loc[idx, 'context']
        
        # Previous index (if exists and is within the same user group)
        if idx - 1 in group.index and pd.isna(data_wo_test.loc[idx - 1, 'context']):
            updates[idx - 1] = context_val
        
        # Next index (if exists and is within the same user group)
        if idx + 1 in group.index and pd.isna(data_wo_test.loc[idx + 1, 'context']):
            updates[idx + 1] = context_val

# Apply updates
data_spreaded_context = deepcopy(data_wo_test)

for idx, val in updates.items():
    data_spreaded_context.at[idx, 'context'] = val

# Check how it looks, for example:
print(data_wo_test.index[data_wo_test['context'].notna()].tolist())
print(data_wo_test.loc[69713:69717])
print(data_spreaded_context.loc[69713:69717])
print(data_wo_test.loc[8102443:8102447])
print(data_spreaded_context.loc[8102443:8102447])
#


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

data_filtered = data_spreaded_context[data_spreaded_context['context'].notna()]

y = data_filtered['context']
y = pd.DataFrame(y)
X = data_filtered.drop('context', axis=1)

y_test = test_set['context']
y_test = pd.DataFrame(y_test)
x_test = test_set.drop('context', axis=1)

x_processed = preprocessor.fit_transform(X)
x_test_processed = preprocessor.fit_transform(x_test)

onehot = OneHotEncoder(sparse_output=False)
y_cat = y.astype('category')
y_uint =  y_cat["context"].cat.codes.astype('uint8')

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
model.add(layers.GRU(500, input_shape=(1,72)))
keras.layers.Dropout(0.2)
#keras.layers.ConvLSTM2D(500, 5)
#keras.layers.SimpleRNN(20)
model.add(layers.BatchNormalization())
keras.layers.Dropout(0.2)
model.add(layers.Dense(20))
keras.layers.Dropout(0.2)
model.add(layers.Dense(4))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	

model.fit(
    x_train_3D, y_train, validation_data=(x_validate_3D, y_validate), batch_size=64, epochs=40
)

results = model.evaluate(x_test_3D, y_test)


