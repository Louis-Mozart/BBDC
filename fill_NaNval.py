import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



class Filler:
  '''This class allows for filling in missing values of all features using polynomial interpolation'''
    
    def fill_all_Id(self, data, deg_ppg, deg_hr,  deg_hrIbi, deg_x, deg_y, deg_z):
        
        print(f"filling the missing values of ppg with a {deg_ppg} degree polynomial")
        ppg_fill = self.fill_all_ppg_Id(data, deg_ppg)
        print("done")

        print(f"filling the missing values of hr with a {deg_hr} degree polynomial")
        hr_fill = self.fill_all_hr_Id(ppg_fill, deg_hr)
        print("done")

        print(f"filling the missing values of hrIbi with a {deg_hrIbi} degree polynomial")
        hrIbI_fill = self.fill_all_hrIbI_Id(hr_fill, deg_hrIbi)
        print("done")

        print(f"filling the missing values of x with a {deg_x} degree polynomial")
        x_fill = self.fill_all_x_Id(hrIbI_fill, deg_x)
        print("done")

        print(f"filling the missing values of y with a {deg_y} degree polynomial")
        y_fill = self.fill_all_y_Id(x_fill, deg_y)
        print("done")

        print(f"filling the missing values of z with a {deg_z} degree polynomial")
        z_fill = self.fill_all_z_Id(y_fill, deg_z)
        print("done")

        return z_fill

    def fill_all_ppg_Id(self, data, deg):
      ''''''
        Ids = data["sessionId"].unique()
        filled_data = []
        for Id in Ids:
            #Fill ppg values
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
    
    def fill_all_x_Id(self, data, deg):

        Ids = data["sessionId"].unique()
        filled_data = []

        for Id in Ids:

            filled_x_data = self.fill_x_val(data, Id, deg)
            filled_data.append(filled_x_data)

        return  pd.concat(filled_data)
    
    def fill_all_y_Id(self, data, deg):

        Ids = data["sessionId"].unique()
        filled_data = []

        for Id in Ids:

            filled_y_data = self.fill_y_val(data, Id, deg)
            filled_data.append(filled_y_data)

        return  pd.concat(filled_data)
    
    def fill_all_z_Id(self, data, deg):

        Ids = data["sessionId"].unique()
        filled_data = []

        for Id in Ids:

            filled_z_data = self.fill_z_val(data, Id, deg)
            filled_data.append(filled_z_data)

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
    
    def fill_x_val(self,data,Id, deg):

        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['x'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['x'], deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['x_filled'] = df['x']
        df.loc[df['x_filled'].isna(), 'x_filled'] = filled_values[df['x_filled'].isna()]
        
        
        return df
    
    def fill_y_val(self,data,Id, deg):

        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['y'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['y'], deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['y_filled'] = df['y']
        df.loc[df['y_filled'].isna(), 'y_filled'] = filled_values[df['y_filled'].isna()]
        
        
        return df
    
    def fill_z_val(self,data,Id, deg):

        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['z'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['z'], deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['z_filled'] = df['z']
        df.loc[df['z_filled'].isna(), 'z_filled'] = filled_values[df['z_filled'].isna()]
        
        
        return df
    

    def fill_hrStatus_val(self,df):

        '''This function predicts the hr_status using the hr_filled column obtained as the filled missing 
            values of the hr column. First, run the fill_all_hr_Id function before running this one.
            The output data is a data containing the filled hrStatus at every row, with the notification and 
            engagement column deleted from the data as well.'''
        
        
        print("filling the missing values of hr_satus with a logistic regression")

        class_mapping = {1.0: 'Class1', -10.0: 'Class2', 0.0: 'Class3', -3.0: 'Class4',
                        -99.0: 'Class5', -999.0: 'Class6', -1.0: 'Class7', -11.0: 'Class8'}
        
        df['hr_status_class'] = df['hrStatus'].map(class_mapping)

        # Split data into complete and incomplete observations
        complete_data = df.dropna(subset=['hr_filled', 'hr_status_class']) #Used the existing values for training
        incomplete_data = df[df['hr_status_class'].isna()]            #used the NaN for prediction

        
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
        incomplete_data['hr_status_filled'] = incomplete_data['hr_status_class_imputed'].map(class_mapping_reverse)


        # Compute the accuracy of the predictor
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("hr_status predicted and filled with an accuracy of:", accuracy)

        #merge the two datasets to get the full data back
        merged = pd.concat([complete_data, incomplete_data])
        merged.drop(["hr_status_class","hr_status_class_imputed","notification","engagement"],axis=1,inplace=True)
        merged['hr_status_filled'] = merged['hr_status_filled'].fillna(merged['hrStatus'])
        
        return merged
    


class refiner:

    ### Merge data


    def merger(Data, prof_skeleton, Data2):  # Data2 = pd.read_csv("SessionData-all.csv")

        # Data = pd.read_csv("prof_data.csv")
        # prof_skeleton = pd.read_csv("prof_skeleton.csv")
        skeleton_merger = prof_skeleton[["sessionId", "timestamp"]]
        missing_columns = ['x', 'y', 'z', 'ppgValue', 'hr', 'hrIbi', 'hrStatus', 
                        'ibiStatus', 'notification', 'engagement', 'affect', 'context']
        for column in missing_columns:
            skeleton_merger[f"{column}"] = np.nan
        Data_merged = pd.concat([Data, skeleton_merger])
        len(Data)+len(prof_skeleton)==len(Data_merged)

        ####### Do interpolation

        # ... Insert your interpolation procedure here
        ppg_filler = Filler()
        Data_merged = ppg_filler.fill_all_Id(Data_merged, deg_ppg= 3, deg_hr= 2, deg_hrIbi= 2, deg_x=1, deg_y=1, deg_z= 1)

        ###### Select prof_skeleton rows and rows with available context/affect

        # Select skeleton data

        # Data_merged['combined'] = Data_merged['timestamp'].astype(str) + '_' + Data_merged['sessionId'].astype(str)
        # prof_skeleton['combined'] = prof_skeleton['timestamp'].astype(str) + '_' + prof_skeleton['sessionId'].astype(str)


        clean_data = ppg_filler.fill_hrStatus_val(Data_merged)

        # Data2 = pd.read_csv("SessionData-all.csv")
        Data2 = Data2.rename(columns={"id":"sessionId"})
        full_data = pd.merge(clean_data, Data2[["sessionId","age","gender","fairNumber"]], on="sessionId")
        #Drop the already interpolated values
        full_data.drop(["x","y","z","ppgValue","hr","hrIbi","ibiStatus","hrStatus"],axis=1,inplace=True)



        ### Complete the test data ####

        Data_merged1 = full_data.set_index(['sessionId', 'timestamp'], inplace=False)

        # Extract supplement information from Data1 using the index of prof_skeleton
        supplement_info = Data_merged1.loc[prof_skeleton.set_index(['sessionId', 'timestamp']).index]

        # Reset the index to make 'ID' and 'timestamp' columns again
        supplement_info.reset_index(inplace=True)

        # Print the supplement information
        supplement_info.drop_duplicates(subset=['sessionId', 'timestamp'], keep='first', inplace=True)


       
        return full_data, supplement_info
    


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


    def remove_nan(full_data):

        data = full_data[full_data['affect'].notna()|full_data['context'].notna()]
        
        return data



    def final_step(data):

        # data = full_data # pd.read_csv("prof_data.csv")

        print("----- Now we are trying  to spread affect variables as some instances are so small ------")
        test_set_affect, training_set_affect = refiner.split_non_na_test_set(data, 20, "affect")

        
        training_set_spreaded_affect = training_set_affect.groupby('sessionId').apply(
        refiner.add_neighbor_values, "affect", "timestamp").reset_index(drop=True)

        print("----- Now we are trying to spread context variables for the data to be balance ------")

        test_set_context, training_set_context = refiner.split_non_na_test_set(data, 20, "context")

        training_set_spreaded_context = training_set_context.groupby('sessionId').apply(
        refiner.add_neighbor_values, "context", "timestamp").reset_index(drop=True)




        training_set_affect = refiner.remove_nan(training_set_spreaded_affect)
        training_set_context = refiner.remove_nan(training_set_spreaded_context)

        subset_data = pd.concat([training_set_affect, training_set_context], ignore_index=True)


        return subset_data
    
    def complete_test_data(data):

        pass
