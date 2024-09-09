import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector

# Set up the Streamlit page
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title('Car Price Prediction')

# Sidebar for user input
st.sidebar.header('Enter Your Expected Car Features')

# Function to take user input
def user_input_features():
    max_power = st.sidebar.number_input('Max Power (in bhp)', min_value=0, value=100, help="Enter the maximum power of the car engine in brake horsepower (bhp).")

    torque = st.sidebar.number_input('Torque (in Nm)', min_value=0, value=200, help="Enter the torque of the car engine in Newton meters (Nm).")

    engine_displacement = st.sidebar.number_input('Engine Displacement (in cc)', min_value=0, value=1500, help="Enter the engine displacement in cubic centimeters (cc).")

    comfort_convenience = st.sidebar.number_input('Comfort & Convenience Rating', min_value=0, max_value=30, value=5, help="Rate the comfort and convenience features on a scale from 0 to 30.")

    wheel_size = st.sidebar.selectbox('Wheel Size (in inches)', options=[13, 14, 15, 16, 17, 18], index=2, help="Select the wheel size of the car in inches.")

    safety = st.sidebar.number_input('Safety Rating', min_value=0, max_value=30, value=3, help="Rate the safety features on a scale from 0 to 30.")

    exterior = st.sidebar.number_input('Exterior Rating', min_value=0, max_value=30, value=4, help="Rate the exterior features on a scale from 0 to 30.")

    interior = st.sidebar.number_input('Interior Rating', min_value=0, max_value=30, value=7, help="Rate the interior features on a scale from 0 to 30.")

    entertainment_communication = st.sidebar.number_input('Entertainment & Communication Rating', min_value=0, value=6, help="Rate the entertainment and communication features on a scale from 0 to 10.")

    body_type = st.sidebar.selectbox('Body Type', options=['SUV', 'Sedan', 'Hatchback'], index=0, help="Select the type of body the car has.")

    model_year = st.sidebar.number_input('Model Year', min_value=2000, max_value=2024, value=2022, help="Enter the year the car model was released.")

    # Encode body type
    body_type_encoded = {'SUV': 0, 'Sedan': 1, 'Hatchback': 2}.get(body_type, 0)

    # Create a DataFrame for the features
    features = {
        'Max Power': max_power,
        'Torque': torque,
        'Engine Displacement': engine_displacement,
        'Comfort & Convenience': comfort_convenience,
        'Wheel Size': wheel_size,
        'Safety': safety,
        'Exterior': exterior,
        'Interior': interior,
        'Entertainment & Communication': entertainment_communication,
        'bt': body_type_encoded,
        'modelYear': model_year
    }
    
    return pd.DataFrame([features])

# Load the model and scaler from pickle files
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scalar_file:
    scaler = pickle.load(scalar_file)

# Function to get database connection
def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='cars'
    )
    return conn

# Get user input features
df = user_input_features()

# Create a layout for the sidebar with centered button
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    predict_button = st.button('Predict')

# Prediction
if predict_button:
    # Scale the features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)
    
    # Convert prediction to Indian Rupees format
    def format_price(price):
        if price >= 10000000:
            return f'₹{price / 10000000:.2f} Crores'
        elif price >= 100000:
            return f'₹{price / 100000:.2f} Lakhs'
        else:
            return f'₹{price:.2f}'

    # Show the formatted predicted price
    st.write(f'Predicted Price: {format_price(prediction[0])}')

# Example historical or full dataset (replace df_full with actual dataset)
df_full = pd.DataFrame({
    'Max Power': [100, 150, 200, 250, 300],
    'Torque': [200, 250, 300, 350, 400],
    'Engine Displacement': [1000, 1500, 2000, 2500, 3000],
    'Comfort & Convenience': [5, 6, 7, 8, 9],
    'Wheel Size': [13, 14, 15, 16, 17],
    'Safety': [3, 4, 5, 6, 7],
    'Exterior': [4, 5, 6, 7, 8],
    'Interior': [7, 8, 9, 10, 11],
    'Entertainment & Communication': [6, 7, 8, 9, 10],
    'bt': [0, 1, 2, 0, 1],
    'modelYear': [2018, 2019, 2020, 2021, 2022],
    'Price': [500000, 600000, 700000, 800000, 900000]
})

# Sidebar for selecting plot type
st.sidebar.header('Visualization Options')
plot_type = st.sidebar.selectbox('Select Plot Type', ['Scatter Plots', 'Pair Plot'])

# Function to plot scatter plots
def plot_scatter_plots(df, feature, target='Price'):
    plt.figure(figsize=(6, 4))  # Smaller size for better fit
    sns.scatterplot(x=df[feature], y=df[target])
    plt.title(f'{target} vs {feature}')
    plt.xlabel(feature)
    plt.ylabel(target)
    st.pyplot(plt)

# Function to plot pair plots
def plot_pair_plot(df):
    plt.figure(figsize=(8, 6))  # Smaller size for better fit
    sns.pairplot(df, x_vars=df.columns[:-1], y_vars=['Price'], height=2.5, aspect=1.2)
    plt.title('Pair Plot of Features vs Price')
    st.pyplot(plt)

# Plotting based on user selection
if plot_type == 'Scatter Plots':
    feature = st.sidebar.selectbox('Select Feature for Scatter Plot', df_full.columns[:-1])
    plot_scatter_plots(df_full, feature)
elif plot_type == 'Pair Plot':
    plot_pair_plot(df_full)

# Feedback section moved to the main page below the graph
st.header('Feedback')
feedback = st.text_area("Please leave your feedback here", height=100)

if st.button('Submit Feedback'):
    if feedback:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (feedback_text) VALUES (%s)", (feedback,))
        conn.commit()
        conn.close()
        st.success("Thank you for your feedback!")
    else:
        st.warning("Please enter some feedback before submitting.")

# Custom CSS styling
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #FF5722; /* Orange background */
        color: white; /* White text */
    }
    .stButton > button:hover {
        background-color: #E64A19; /* Darker orange on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)
