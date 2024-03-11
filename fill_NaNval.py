import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



class Filler:
    # def __init__(self, deg):
    #     self.deg = deg


    def fill_all_Id(self, data, deg_ppg, deg_hr,  deg_hrIbi):

        print(f"filling the missing values of ppg with a {deg_ppg} degree polynomial")
        ppg_fill = self.fill_all_ppg_Id(data, deg_ppg)
        print("done")

        print(f"filling the missing values of hr with a {deg_hr} degree polynomial")
        hr_fill = self.fill_all_hr_Id(ppg_fill, deg_hr)
        print("done")

        print(f"filling the missing values of hrIbi with a {deg_hrIbi} degree polynomial")
        hrIbI_fill = self.fill_all_hrIbI_Id(hr_fill, deg_hrIbi)
        print("done")

        return hrIbI_fill

    def fill_all_ppg_Id(self, data, deg):

        Ids = data["sessionId"].unique()
        filled_data = []

        for Id in Ids:

             # Fill ppg values
            filled_ppg_data = self.fill_ppg_val(data, Id, deg)
            filled_data.append(filled_ppg_data)

        return  pd.concat(filled_data)


    def fill_all_hr_Id(self, data, deg):

        Ids = data["sessionId"].unique()
        filled_data = []

        for Id in Ids:

            filled_ppg_data = self.fill_hr_val(data, Id, deg)
            filled_data.append(filled_ppg_data)

        return  pd.concat(filled_data)
    
    def fill_all_hrIbI_Id(self, data, deg):

        Ids = data["sessionId"].unique()
        filled_data = []

        for Id in Ids:

            filled_hrIbi_data = self.fill_hrIbI_val(data, Id, deg)
            filled_data.append(filled_hrIbi_data)

        return  pd.concat(filled_data)


    def fill_ppg_val(self, data, Id, deg):
        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['ppgValue'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['ppgValue'], deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['ppg_filled'] = df['ppgValue']
        df.loc[df['ppg_filled'].isna(), 'ppg_filled'] = filled_values[df['ppg_filled'].isna()]

        return df
    

    def fill_hr_val(self,data,Id, deg):

        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['hr'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['hr'], deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['hr_filled'] = df['hr']
        df.loc[df['hr_filled'].isna(), 'hr_filled'] = filled_values[df['hr_filled'].isna()]
        
        
        return df
    
    def fill_hrIbI_val(self,data,Id, deg):

        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['hrIbi'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['hrIbi'], deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['hrIbi_filled'] = df['hrIbi']
        df.loc[df['hrIbi_filled'].isna(), 'hrIbi_filled'] = filled_values[df['hrIbi_filled'].isna()]
        
        
        return df
    

    def fill_hrStatus_val(self,df):

        '''This function predict the hr_status using the hr_filled column obtained as the filled missing 
            values of the hr column. So First run the fill_all_hr_Id function before running this function.
            The output data is a data containing the filled hrStatus at every rows. with the notification and 
            engagement column deleted from the data as well.'''
        
        #First map the target data to categorical values for classification task.

        print("filling the missing values og hr_satus with a logistic regression")

        class_mapping = {1.0: 'Class1', -10.0: 'Class2', 0.0: 'Class3', -3.0: 'Class4',
                        -99.0: 'Class5', -999.0: 'Class6', -1.0: 'Class7', -11.0: 'Class8'}

        # Replace numeric values with corresponding classes
        df['hr_status_class'] = df['hrStatus'].map(class_mapping)

        # Split data into complete and incomplete observations
        complete_data = df.dropna(subset=['hr_filled', 'hr_status_class']) #Used the existing values for training
        incomplete_data = df[df['hr_status_class'].isna()]            #used the NaN for prediction

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
        print("The accuracy of our predictor is:", accuracy)

        #merge the two data togeteher to get the full data back
        merged = pd.concat([complete_data, incomplete_data])
        merged.drop(["hr_status_class","hr_status_class_imputed","notification","engagement"],axis=1,inplace=True)
        merged['hr_status_imputed'] = merged['hr_status_imputed'].fillna(merged['hrStatus'])
        
        return merged

