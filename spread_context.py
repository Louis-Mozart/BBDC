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

def split_non_na_test_set(data, test_set_size, variable_name):

    # Identify rows where 'context' is not NaN
    non_na_indices = data.index[data[variable_name].notna()].tolist()
    
    random_selected_indices = np.random.choice(non_na_indices, 
        size=min(test_set_size, len(non_na_indices)), replace=False)
    
    test_set = data.loc[random_selected_indices]
    data_without_test_set = data.drop(random_selected_indices)
    
    return test_set, data_without_test_set
    


def add_neighbor_values(group, target_variable, time_variable):
    # Iterate over rows with a non-NaN context
    for index, row in group[group[target_variable].notna()].iterrows():
        context_val = row[target_variable]
        timestamp_val = row[time_variable]
        
        # Identify rows within the timestamp range and without a context
        mask = (group[time_variable] >= timestamp_val - 2000) & \
               (group[time_variable] <= timestamp_val + 2000) & \
               (group[target_variable].isna())
        
        # Update 'context' for these rows
        group.loc[mask, target_variable] = context_val
        
    return group

if __name__ == "__main__":
        
    data = pd.read_csv("/home/leo/sciebo/BBDC/prof_data.csv")
    
    test_set, training_set = split_non_na_test_set(data, 20, "context")
    
    training_set_spreaded_context = training_set.groupby('sessionId').apply(
        add_neighbor_values, "context", "timestamp").reset_index(drop=True)
    
    #Check some examples
    filtered_df = training_set[training_set['context'].notna()]
    timestamps_and_sessions = filtered_df[['timestamp', 'sessionId']]
    print(timestamps_and_sessions)    
    # Look at an example
    print(training_set.loc[69714:69717])
    subset_for_checking = training_set_spreaded_context[(
        training_set_spreaded_context['timestamp'] >= 2700529 - 2500) 
        & (training_set_spreaded_context['timestamp'] <= 2700529 + 2500)]
    # Another example
    print(training_set.loc[1351460:1351470])
    subset_for_checking2 = training_set_spreaded_context[(
        training_set_spreaded_context['timestamp'] >= 1000405 - 2500) 
        & (training_set_spreaded_context['timestamp'] <= 1000405 + 2500)]
    