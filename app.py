import streamlit as st
from src1.prediction import Insurance_Predictor
st.title("Insurance Premium Prediction")
st.write("Enter the details to predict the insurance premium")
Age = st.number_input("enter age")
Annual_Income_LPA = st.number_input("Annual Income (LPA)")
Policy_Term_Years = st.number_input("Policy Term (Years)")
Sum_Assured_Lakhs = st.number_input("Sum Assured (Lakhs)")
if st.button("Predict"):
    model = Insurance_Predictor()
    result = model.predict(Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs)
    st.success(result)