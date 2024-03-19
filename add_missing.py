import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from fill_NaNval import Filler 
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")



Data1 = pd.read_csv("prof_data.csv")
Data1.head()

ppg_filler = Filler()
Data1 = ppg_filler.fill_all_Id(Data1, deg_ppg= 3, deg_hr= 6, deg_hrIbi= 2, deg_x=1, deg_y=1, deg_z= 1)

clean_data = ppg_filler.fill_hrStatus_val(Data1)
clean_data.tail()

Data2 = pd.read_csv("SessionData-all.csv")
Data2 = Data2.rename(columns={"id":"sessionId"})
full_data = pd.merge(clean_data, Data2[["sessionId","age","gender","fairNumber"]], on="sessionId")
#Drop the already interplated values
full_data.drop(["x","y","z","ppgValue","hr","hrIbi","ibiStatus","hrStatus"],axis=1,inplace=True)

Data1 = clean_data.copy()
df =  Data1[Data1['sessionId'] == 15]

plt.figure(figsize=(20, 6))
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['hrIbi'], marker='.', linestyle='-', color='red', label='Original hr')
plt.title('Original Heart Rate over Time')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Heart Rate (bpm)')
plt.grid(True)
plt.legend()

# Create the second subplot for the interpolated 'ppg_filled'
plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['hrIbi_filled'], marker='.', linestyle='-', color='blue', label='Interpolated hr_filled')
plt.title('Interpolated Heart Rate over Time')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Heart Rate (bpm)')
plt.grid(True)
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming df is your DataFrame containing 'hr' and 'hr_status' columns
# Map numeric values to classes
class_mapping = {1.0: 'Class1', -10.0: 'Class2', 0.0: 'Class3', -3.0: 'Class4',
                 -99.0: 'Class5', -999.0: 'Class6', -1.0: 'Class7', -11.0: 'Class8'}

# Replace numeric values with corresponding classes
df['hr_status_class'] = df['hrStatus'].map(class_mapping)

# Split data into complete and incomplete observations
complete_data = df.dropna(subset=['hr_filled', 'hr_status_class'])
incomplete_data = df[df['hr_status_class'].isna()]

# Prepare data for classification
X = complete_data[['hr_filled']]
y = complete_data['hr_status_class']

# Train logistic regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict missing 'hr_status' values
incomplete_data['hr_status_class_imputed'] = model.predict(incomplete_data[['hr_filled']])

# Map predicted classes back to numeric values
class_mapping_reverse = {v: k for k, v in class_mapping.items()}
incomplete_data['hr_status_imputed'] = incomplete_data['hr_status_class_imputed'].map(class_mapping_reverse)


# Compute accuracy of the predictor
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the predictor:", accuracy)

merged = pd.concat([complete_data, incomplete_data])
merged.drop(["hr_status_class","hr_status_class_imputed","notification","engagement"],axis=1,inplace=True)
merged['hr_status_imputed'] = merged['hr_status_imputed'].fillna(merged['hrStatus'])
merged

# Check imputed data
df.head()

clean_data[clean_data["ibiStatus"]==0&clean_data["hrIbi"].notna()]

#PPGvalue test.
df =  Data1[Data1['sessionId'] == 8]

plt.figure(figsize=(20, 6))
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['ppgValue'], marker='.', linestyle='-', color='red', label='Original ppgValue')
plt.title('Original Heart Rate over Time')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Heart Rate (bpm)')
plt.grid(True)
plt.legend()

# Create the second subplot for the interpolated 'ppg_filled'
plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['ppg_filled'], marker='.', linestyle='-', color='blue', label='Interpolated ppg_filled')
plt.title('Interpolated Heart Rate over Time')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Heart Rate (bpm)')
plt.grid(True)
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()