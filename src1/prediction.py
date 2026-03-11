#1. load scaler.pkl and model.pkl files
#2. create a function to predict
import pickle
import numpy as np
class Insurance_Predictor:
    def __init__(self):
        with open('artifacts/model.pkl','rb') as f:
            self.model=pickle.load(f)
        with open('artifacts/scaler.pkl','rb') as f:
            self.scaler=pickle.load(f)

    def predict(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input_data=np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        input_data_scaled=self.scaler.transform(input_data)
        prediction=self.model.predict(input_data_scaled)
        return prediction[0]