import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
column_names = joblib.load('columns.pkl')  # Load the saved column names used during training

# Title and Description with enhanced styling
st.title('🏠 **House Price Prediction Using Machine Learning**')
st.markdown("""
    This app predicts the price of a house based on various features such as area, year built, and more.
    **Fill in the details** and get an estimate of the house price!
""", unsafe_allow_html=True)

# Sidebar Styling
st.sidebar.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #f4f6f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-header {
            font-size: 22px;
            font-weight: bold;
            color: #2d9cdb;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar header
st.sidebar.header('Input Features')

# Input fields for categorical and numerical features
MSSubClass = st.sidebar.selectbox("MSSubClass (Type of Dwelling)", [20, 30, 40, 50, 60, 70, 80, 90])
MSZoning = st.sidebar.selectbox("MSZoning (Zoning Classification)", ['RL', 'RM', 'FV', 'C (all)', 'I (all)'])
LotArea = st.sidebar.number_input("LotArea (Lot Size in sq ft)", min_value=1000, max_value=100000)
LotConfig = st.sidebar.selectbox("LotConfig (Lot Configuration)", ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
BldgType = st.sidebar.selectbox("BldgType (Building Type)", ['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'TwnhsI'])
OverallCond = st.sidebar.slider("OverallCond (Condition of House)", 1, 10, 5)
YearBuilt = st.sidebar.number_input("YearBuilt (Year of Construction)", min_value=1900, max_value=2025)
YearRemodAdd = st.sidebar.number_input("YearRemodAdd (Year of Remodel)", min_value=1900, max_value=2025)
Exterior1st = st.sidebar.selectbox("Exterior1st (Exterior Material)", ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock'])
BsmtFinSF2 = st.sidebar.number_input("BsmtFinSF2 (Finished Basement Area)", min_value=0, max_value=5000)
TotalBsmtSF = st.sidebar.number_input("TotalBsmtSF (Total Basement Area)", min_value=0, max_value=5000)

# Combine all inputs into a DataFrame
input_data = {
    'MSSubClass': [MSSubClass],
    'MSZoning': [MSZoning],
    'LotArea': [LotArea],
    'LotConfig': [LotConfig],
    'BldgType': [BldgType],
    'OverallCond': [OverallCond],
    'YearBuilt': [YearBuilt],
    'YearRemodAdd': [YearRemodAdd],
    'Exterior1st': [Exterior1st],
    'BsmtFinSF2': [BsmtFinSF2],
    'TotalBsmtSF': [TotalBsmtSF]
}

# Convert input data into DataFrame
input_df = pd.DataFrame(input_data)

# One-Hot Encoding of categorical features
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Align input data columns with model columns
input_encoded = input_encoded.reindex(columns=column_names, fill_value=0)

# Define the predict button with a modern touch
if st.button('🔮 **Predict House Price**'):
    with st.spinner('Predicting...'):
        # Apply the same scaling used in training
        input_scaled = scaler.transform(input_encoded)

        # Make prediction using the trained model
        prediction = model.predict(input_scaled)

        # Display predicted result in a modern card container
        st.markdown(f"""
            <div style="padding: 20px; background-color: #f7f8fa; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="text-align:center; color:#2D9CDB;">Predicted House Price: <span style="color: #27AE60;">${prediction[0]:,.2f}</span></h3>
            </div>
        """, unsafe_allow_html=True)

# Additional Styling for Inputs and Buttons
st.markdown("""
    <style>
        .stButton>button {
            background-color: #27AE60;
            color: white;
            font-size: 18px;
            border-radius: 5px;
            width: 100%;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #2D9CDB;
        }
        .css-ffhzg2 {
            font-size: 18px;
            padding: 10px;
            background-color: #f7f8fa;
            border-radius: 10px;
        }
        .stSlider>div {
            font-size: 16px;
        }
        .stTextInput>div>input {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)
