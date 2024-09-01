# Car Price Prediction 

## Overview
This project aims to predict car prices based on various features such as 'Model Year', 'Seats', 'Number of Owners', 'Kms Driven', 'Engine Displacement', and 'Mileage'. The workflow includes data transformation, cleaning, model training, and deployment using Streamlit.

## Project Workflow

1. **Unstructured Data Conversion**
   - The initial dataset was provided in an unstructured JSON format.
   - The data was first converted into a structured format by flattening the JSON data.
   - The structured data is saved in the "structured data" folder.

2. **Data Cleaning Process**
   - The structured data underwent a thorough cleaning process to handle missing values, remove duplicates, and ensure consistency across all features.
   - The cleaned data is saved in the "Data Cleaning process" folder.

3. **Model Selection and Testing**
   - Various machine learning algorithms were tested to find the best fit for the car price prediction task.
   - The XGBoost algorithm was chosen for its superior performance in terms of accuracy and efficiency.

4. **Model Deployment**
   - The final model was deployed using Streamlit, a Python framework for creating interactive web applications.
   - The application predicts the target variable 'car price' based on the provided features.

## Features Used in Prediction
- **Model Year**: The year the car model was manufactured.
- **Seats**: The number of seats in the car.
- **Number of Owners**: The number of previous owners the car has had.
- **Kms Driven**: The total kilometers driven by the car.
- **Engine Displacement**: The engine's displacement in cubic centimeters (cc).
- **Mileage**: The fuel efficiency of the car, typically measured in km/l or mpg.

## Files and Directories
- **structured data/**: Contains the structured and flattened dataset.
- **Data Cleaning process/**: Contains the cleaned dataset ready for model training.
- **Model/**: Contains the trained XGBoost model and related files.
- **Streamlit App/**: Contains the Streamlit application code for car price prediction.

## Usage
1. **Prepare the Environment**: Install necessary Python libraries using `pip install -r requirements.txt`.
2. **Run the Streamlit App**: Navigate to the "Streamlit App" directory and run `streamlit run app.py`.
3. **Input Features**: Enter the relevant car details to predict the price.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`

## Conclusion
The Car Price Prediction project demonstrates the process of transforming unstructured data, cleaning and preprocessing it, and using a machine learning model for price prediction. The XGBoost algorithm provided reliable predictions, and the model was successfully deployed in a user-friendly Streamlit application.
