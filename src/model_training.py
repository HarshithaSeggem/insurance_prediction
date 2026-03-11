#load the processed data from processed folder
#2 create modeland train data
#3.save model in model folder
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
X_train=pd.read_csv('../data/processed/x_train.csv')
y_train=pd.read_csv('../data/processed/y_train.csv')
y_train=pd.read_csv('../data/processed/y_train.csv')
X_test=pd.read_csv('../data/processed/x_test.csv')
print(X_train)
model = LinearRegression()
model.fit(X_train,y_train)
with open('../artifacts/model.pkl','wb') as f:
    pickle.dump(model,f)