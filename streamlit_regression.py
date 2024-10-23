import streamlit as st
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import load_model

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = load_model('model.h5')


with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scalar = pickle.load(file)

st.title('Estimated Salary Prediction')


geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age =st.slider('Age', 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
tenure = st.slider('tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_Cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0,1])
is_exited = st.selectbox('Is exited', [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': geography,
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_Cr_card],
    'IsActiveMember': [is_active_member],
    'Exited' : [is_exited]
})

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = input_data.drop(['Geography'], axis=1)


input_data_scaled = scalar.transform(input_data)

prediction = model.predict(input_data_scaled)


st.write('Estimated salary of the customer:', prediction)
