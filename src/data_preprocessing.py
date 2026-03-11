import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data():
    data = pd.read_csv("../data/raw/insuranceprediction.csv")

    X = data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y = data['Annual_Premium_Thousands']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_and_split_data()

print("Data loaded successfully")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
