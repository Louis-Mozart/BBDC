import numpy as np
import pandas as pd



class Filler:
    def __init__(self, deg):
        self.deg = deg

    def fill_all_Id(self, data):
        Ids = data["sessionId"].unique()
        filled_data = []
        for Id in Ids:
            filled_data.append(self.fill_ppg_val(data, Id))
        return pd.concat(filled_data)

    def fill_ppg_val(self, data, Id):
        df = data[data["sessionId"]==Id]
        df_clean = df.dropna(subset=['ppgValue'])  
        poly_coeffs = np.polyfit(df_clean['timestamp'], df_clean['ppgValue'], self.deg)
        filled_values = np.poly1d(poly_coeffs)(df['timestamp'])
        df['ppg_filled'] = df['ppgValue']
        df.loc[df['ppg_filled'].isna(), 'ppg_filled'] = filled_values[df['ppg_filled'].isna()]
        return df
    

    def fill_hr_val(self,data,Id):

        pass
