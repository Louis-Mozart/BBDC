import numpy as np
import pandas as pd



class Filler:
    # def __init__(self, deg):
    #     self.deg = deg


    def fill_all_Id(self, data, deg_ppg, deg_hr):

        ppg_fill = self.fill_all_ppg_Id(data, deg_ppg)
        hr_fill = self.fill_all_hr_Id(ppg_fill, deg_hr)

        return hr_fill

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

