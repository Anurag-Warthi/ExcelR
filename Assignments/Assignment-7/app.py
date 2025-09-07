import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

# Load the trained model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Titanic Survival Prediction")

# Collect user inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert categorical inputs
sex = 1 if sex == "Female" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# Prepare input data
input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked_q, embarked_s]],
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"])

# Add constant term (as done during model training)
input_data = sm.add_constant(input_data)

# Predict survival
if st.button("Predict"):
    prediction = model.predict(input_data)
    survival = "Survived" if prediction[0] >= 0.5 else "Did Not Survive"
    st.write(f"Prediction: **{survival}**")