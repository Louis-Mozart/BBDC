import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



# This is work in progress and does not run yet




data = pd.read_csv("/home/leo/sciebo/BBDC/bbdcdata_10000.csv")
print(list(data.columns))
numeric_features = ['timestamp', 'ppg_filled', 
                    'hr_filled', 'hrIbi_filled', 'x_filled', 'y_filled', 
                    'z_filled']
categorical_features = ['sessionId', 'affect', 'context', 'hr_status_filled', 
                        'age', 'gender', 'fairNumber']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_filtered = data[data['context'].notna()]

data_processed = preprocessor.fit_transform(data_filtered)

# TODO separate X and y

test_size = 10  # Number of test samples
validation_size = 0.2  # Proportion of validation set size relative to (training + validation)

test_proportion = test_size / len(X)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_proportion, random_state=42)

X_train, X_validate, y_train, y_validate = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

array = X.to_numpy()

print(data.shape)



model = keras.Sequential()
model.add(layers.GRU(64, input_shape=14))
model.add(layers.BatchNormalization())
model.add(layers.Dense(4))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)	

model.fit(
    x_train, y_train, validation_data=(x_validate, y_validate), batch_size=64, epochs=10
)

for i in range(10):
    result = tf.argmax(model.predict(tf.expand_dims(x_test[i], 0)), axis=1)
    print(result.numpy(), y_test[i])
    
