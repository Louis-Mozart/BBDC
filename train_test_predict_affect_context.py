import numpy as np
from fill_NaNval import refiner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

    
class Modeller_affect_context: 
    '''Preprocess the data for training with the steps 1--4
        1. Merge the data
        2. Encode
        3. Train
        4. Evaluate
    '''
    
    def __init__(self,Data1, Data2, skeleton):

        """"  
            Data2 = pd.read_csv("SessionData-all.csv")
            Data1 = pd.read_csv("prof_data.csv")
            skeleton = pd.read_csv("prof_skeleton.csv")      
        """

        self.label_encoder = LabelEncoder()
        self.label_encoder_test = LabelEncoder()
        self.Data1 = Data1
        self.Data2 = Data2
        # self.Data3 = Data3
        self.skeleton = skeleton
        
        print("------ Preprocessing starts now ------------")

        self.full_data, self.Data3  = refiner.merger(Data = self.Data1, prof_skeleton = self.skeleton, Data2 = self.Data2)

        self.new_data = self.encode_test_data(self.Data3)

        self.subset_data = refiner.final_step(self.full_data)

        print("------ Preprocessing done now ------------")

    def encode_test_data(self,subset_data):

        '''This function encodes age and gender variables to integer values on the test data'''

        affect_data = subset_data

        affect_data['age_enc'] = self.label_encoder_test.fit_transform(affect_data['age'])
        affect_data['gender_enc'] = self.label_encoder_test.fit_transform(affect_data['gender'])

        new_data = affect_data[['sessionId','timestamp','ppg_filled', 'hr_filled', 'hrIbi_filled', 'x_filled', 'y_filled', 'z_filled', 'hr_status_filled','fairNumber','age_enc', 'gender_enc']].values

    
        return  new_data
    


    def encode_affect_context(self, subset_data, word):
        '''Encode affect, context, and other categorical variables to integer values'''

        affect_data = subset_data[subset_data[word].notna()]
        affect_data['age_enc'] = self.label_encoder.fit_transform(affect_data['age'])
        affect_data['gender_enc'] = self.label_encoder.fit_transform(affect_data['gender'])
        affect_data[f'{word}_enc'] = self.label_encoder.fit_transform(affect_data[word])

        return affect_data

    def train_predict_affect(self): 

        #sorry for this :( was running out of time. Will fix later 
        skeleton = self.skeleton
        subset_data = self.subset_data
        affect_data = self.encode_affect_context(subset_data,"affect")
        X = affect_data[['sessionId','timestamp','ppg_filled', 'hr_filled', 'hrIbi_filled', 'x_filled', 'y_filled', 'z_filled', 'hr_status_filled','fairNumber','age_enc', 'gender_enc']]
        y = affect_data['affect_enc']
        np.random.seed(18) 

        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models (RFs or DTs work better)
        models = [
            RandomForestClassifier(),
            # MLPClassifier(class_weight=dict(zip(np.unique(y_train), class_weights)))
            # SVC(class_weight=dict(zip(np.unique(y_train), class_weights)))
        ]




        # Train and evaluate each model

        for model in models:
            # Train the model
            print(f'---------------- training start for affect variables with a {type(model).__name__} model --------------------------')
            model.fit(X_train, y_train)
            
            # Evaluate the model with cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Calculate the mean accuracy across all folds
            mean_accuracy = np.mean(scores)
            
            # Print the mean accuracy
            print(f"Model: {type(model).__name__}, Mean Accuracy: {mean_accuracy:.5f}")
            
            # Optionally, you can also evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = float(classification_report(y_test, y_pred, output_dict=True)['accuracy'])
            print(f"Accuracy on test set: {accuracy:.5f}")
            # print(classification_report(y_test, y_pred))

            print(np.unique(y_pred))
        

        predicted_labels_RF = model.predict(self.new_data)

        print(f' the predicted values of affect on the test data are: {np.unique(predicted_labels_RF)}')

        # skeleton = pd.read_csv("prof_skeleton.csv")
        skeleton["affect"] = self.label_encoder.inverse_transform(predicted_labels_RF)

        print(f'The model was able to identify {skeleton["affect"].unique()} from the skeleton file')

        return skeleton

    

    def train_predict_context(self, skeleton): 
    
        subset_data = self.subset_data

        
        context_data = self.encode_affect_context(subset_data,"context")

        X = context_data[['sessionId','timestamp','ppg_filled', 'hr_filled', 'hrIbi_filled', 'x_filled', 'y_filled', 'z_filled', 'hr_status_filled','fairNumber','age_enc', 'gender_enc']]

        # X = context_data[['x_filled', 'y_filled', 'z_filled','hr_status_filled']]

        y = context_data['context_enc']

        # split the data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models
        models = [
            RandomForestClassifier(),
            # MLPClassifier()
            # SVC()
                ]

        # Train and evaluate each model
        for model in models:
            # Train the model
            print(f'---------------- training start for context variables with a {type(model).__name__} model --------------------------')

            model.fit(X_train, y_train)
            
            # Evaluate the model with cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Calculate the mean accuracy across all folds
            mean_accuracy = np.mean(scores)
            
           
            print(f"Model: {type(model).__name__}, Mean Accuracy: {mean_accuracy:.5f}")
            
            y_pred = model.predict(X_test)
            accuracy = float(classification_report(y_test, y_pred, output_dict=True)['accuracy'])
            print(f"Accuracy on test set: {accuracy:.4f}")

        
        predicted_labels_RF = model.predict(self.new_data)

        print(f' the predicted values of affect on the test data are: {np.unique(predicted_labels_RF)}')

        # skeleton = pd.read_csv("prof_skeleton.csv")
        skeleton["context"] = self.label_encoder.inverse_transform(predicted_labels_RF)

        print(f'The model was able to identify{skeleton["context"].unique()} from the skeleton file')

        return skeleton
