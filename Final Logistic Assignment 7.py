
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pickle
import streamlit as st

st.title('Logistic Regression Model Deployment')
st.write('This app predicts the target variable based on user inputs.')
def user_input_features():
    PassengerId = st.number_input('PassengerId', min_value=1, max_value=1000, value=1)
    Pclass = st.selectbox('Pclass', options=[1, 2, 3], index=0)
    Age = st.number_input('Age', min_value=0.0, max_value=100.0, value=30.0)
    SibSp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
    Parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
    Fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0)
    Sex_male = st.selectbox('Sex_male', options=[True, False], index=0)
    Embarked_Q = st.selectbox('Embarked_Q', options=[True, False], index=0)
    Embarked_S = st.selectbox('Embarked_S', options=[True, False], index=0)
    user_input = pd.DataFrame({
    'PassengerId': [PassengerId],
    'Pclass': [Pclass],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Sex_male': [Sex_male],
    'Embarked_Q': [Embarked_Q],
    'Embarked_S': [Embarked_S]
    })
    return user_input

df=user_input_features()
st.subheader('User Input parameters')
st.write(df)
# load the model from disk
loaded_model =pickle.load(open('Finalized_LogisticRegression_model.pkl', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)



