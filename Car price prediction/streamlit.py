import streamlit as st
import pickle
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# Load the model and scaler
with open('D:\Car price prediction\car_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

with open('D:\Car price prediction\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def predict_price(model, scaler, features):
    # Standardize the features using the loaded scaler
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

def format_price(price):
    # Convert the price to lakhs or crores
    if price >= 10000000:
        return f'₹{price / 10000000:.2f} Crores'
    elif price >= 100000:
        return f'₹{price / 100000:.2f} Lakhs'
    else:
        return f'₹{price:.2f}'

# Streamlit app
st.title('🚗 Car Price Prediction')

# Custom CSS for the main page and sidebar
st.markdown(
    """
    <style>
    /* Main content area styling */
    .main {
        background-color: #f0f0f0;  /* Light gray background for main content */
        color: black;  /* Default text color for main content */
        padding: 20px;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;  /* Set sidebar background to black */
        color: white !important;  /* Default text color for sidebar */
    }
    /* Sidebar header and input labels styling */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] p {
        color: white !important;  /* Ensure sidebar header and labels text color is white */
    }
    /* Sidebar input styling */
    [data-testid="stSidebar"] .stSlider, 
    [data-testid="stSidebar"] .stNumberInput, 
    [data-testid="stSidebar"] .stTextInput {
        color: white !important;  /* Ensure sidebar input labels text color is white */
    }
    /* Button styling */
    .stButton button {
        background-color: #ff0000;  /* Red button */
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;  /* Align button to be full width */
    }
    /* Prediction result styling */
    .prediction {
        font-size: 24px;
        color: #ff6347;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Wrap content in a div with class 'main' for background color styling
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    # Sidebar inputs
    st.sidebar.header('Enter Car Details')
    
    model_year = st.sidebar.slider('Model Year', min_value=2000, max_value=2024, value=2020)
    seats = st.sidebar.slider('Seats', min_value=1, max_value=10, value=4)
    owner_no = st.sidebar.slider('Number of Owners', min_value=1, max_value=5, value=1)
    kms_driven = st.sidebar.number_input('Kms Driven', min_value=0, max_value=200000, value=12000)
    engine_displacement = st.sidebar.number_input('Engine Displacement', min_value=0, max_value=5000, value=1500)
    mileage = st.sidebar.number_input('Mileage', min_value=0, max_value=30, value=20)

    # Collect features
    features = [model_year, seats, owner_no, kms_driven, engine_displacement, mileage]

    # Predict button
    if st.sidebar.button('Predict Price'):
        prediction = predict_price(model, scaler, features)
        formatted_prediction = format_price(prediction)

        # Use components.html to create a styled output
        components.html(
            f"""
            <div class="prediction">
                Predicted Price: {formatted_prediction}
            </div>
            """,
            height=100,
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
