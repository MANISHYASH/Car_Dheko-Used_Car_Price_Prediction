Car Price Prediction Project
----------------------------

Overview
---------
This project aims to predict car prices based on various features extracted from an unstructured JSON dataset. The workflow includes data conversion, cleaning, feature selection, model training, and deployment using Streamlit.

Project Workflow
----------------
1. Unstructured Data Conversion
   - The initial dataset was in an unstructured JSON format, saved in the "DATASET" folder.
   - We converted this JSON data into a structured format by flattening it, and saved the output in the "structured data" folder.

2. Data Cleaning Process
   - The structured data was divided into six files.
   - Each file underwent a detailed cleaning process, including handling missing values, removing duplicates, and standardizing formats. These cleaned files are stored in the "data cleaning" folder.
   - All six files were then merged for further analysis.

3. Feature Selection
   - After analyzing the correlation between various features and the target variable (price), the following features were selected for prediction:
     * Max Power
     * Torque
     * Engine Displacement
     * Comfort & Convenience
     * Wheel Size
     * Safety
     * Exterior
     * Interior
     * Entertainment & Communication
     * bt
     * modelYear

4. Model Selection and Testing
   - Six machine learning models were tested, with the XGBoost algorithm providing the highest accuracy.
   - Hyperparameter tuning was performed to further improve the accuracy of the model.

5. Model Deployment
   - The final XGBoost model was deployed using Streamlit.
   - Two pickle files were created:
     * model.pkl - for the trained XGBoost model.
     * scaler.pkl - for data normalization to preserve the structure during deployment.
   - Feedback form integration was implemented using a database connection via XAMPP server.

Files and Directories
----------------------
- DATASET/: Contains the original unstructured JSON dataset.
- structured data/: Contains the structured, flattened dataset.
- data cleaning/: Contains the cleaned and processed datasets.
- model.pkl and scaler.pkl: Contain the trained XGBoost model and standardization pickle files.
- app.py: Contains the Streamlit application code for car price prediction.

Usage
------
1. Run the Streamlit App:
   - Navigate to the Streamlit App directory and run:  
     `streamlit run app.py`
2. Input Features:
   - Enter the relevant car details in the Streamlit app to predict the car price.

Requirements
-------------
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, xgboost, streamlit

Conclusion
-----------
This project demonstrates how unstructured data can be transformed, cleaned, and used to predict car prices using machine learning. The XGBoost model delivered reliable predictions after hyperparameter tuning, and the model is successfully deployed in a user-friendly Streamlit app, with feedback integrated through XAMPP server.
