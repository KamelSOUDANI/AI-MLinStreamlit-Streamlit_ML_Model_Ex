# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data

file_path=r'D:\Streamlit_ML_Model\Data\Data_BarbeauFlux_NoNaN.csv'
data = pd.read_csv(file_path, header=0, sep=';', decimal='.', index_col=0)
column_names = data.columns.tolist()

# Drop rows with NaN values
data = data.dropna()

# Get column names
column_names = data.columns

# Explanatory text
st.sidebar.markdown('<h6 style="color: black;"> Author : Kamel SOUDANI </h6>', unsafe_allow_html=True)

st.sidebar.markdown('<h4 style="color: black;"> This progam allows to select Output and Input Variables from the list. Note that if there is no selection, an error appears and disappears if output and inputs are selected</h4>', unsafe_allow_html=True)
# Sidebar for user input selection
st.sidebar.markdown('<h1 style="color: blue;">Select Output and Input Variables</h1>', unsafe_allow_html=True)
# Select output variable
output_variable_model = st.sidebar.selectbox('Select Output Variable', column_names)

# Select input variables
input_variables_model = st.sidebar.multiselect('Select Input Variables', column_names, default=['R_450', 'R_550', 'R_650', 'R_720', 'R_750', 'R_800'])

if not output_variable_model or not input_variables_model:
    st.warning('Select Output and Input Variables to start.')

# User option for setting the rate of test data
test_data_rate = st.sidebar.slider('Select the rate of test data (%)', 0, 100, 20, 5)

# Define input features (X) and target variable (y) for model training
X_model = data[input_variables_model]
y_model = data[output_variable_model]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=test_data_rate / 100, random_state=42)

# Train Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)
# Streamlit application
# First page: Model Training and Validation
st.title('Machine Learning Model: Model Training and Validation')

# Display information about the trained model
st.header('Model Information')
st.write(f'Output Variable (Target): {output_variable_model}')
st.write(f'Input Variables: {", ".join(input_variables_model)}')
st.write(f'Training Data Shape: {X_train.shape}')
st.write(f'Test Data Shape: {X_test.shape}')

# Display scatter plot chart of predicted vs observed values for test data
st.subheader('Scatter Plot: Predicted vs Observed (Test Data)')

test_predictions = model.predict(X_test)
scatter_df = pd.DataFrame({'Observed': y_test, 'Predicted': test_predictions})
fig, ax = plt.subplots(figsize=(8, 5))

# Create a scatter plot with a regression line
fig, ax = plt.subplots(figsize=(8, 5))
scatter_plot = sns.regplot(x='Observed', y='Predicted', data=scatter_df, ax=ax)

# Calculate R-squared value
r_squared = r2_score(y_test, test_predictions)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

# Add R-squared and RMSE as annotations
text = f'R-squared: {r_squared:.2f}\nRMSE: {rmse:.2f}'
ax.text(0.05, 0.9, text, transform=ax.transAxes, color='blue', fontsize=12)

# Customize title and labels
scatter_plot.set_title(f'Predicted vs Observed {output_variable_model}', color='blue')
scatter_plot.set_xlabel('Observed', color='blue')
scatter_plot.set_ylabel('Predicted', color='blue')

# Show the plot
st.pyplot(fig)

# Display feature importance chart
st.subheader('Feature Importance Chart')
feature_importance_model = pd.Series(model.feature_importances_, index=input_variables_model).sort_values(ascending=False)
fig, ax=plt.subplots(figsize=(10, 6))
sns.barplot(x=feature_importance_model, y=feature_importance_model.index, palette='viridis')
ax.set_title('Random Forest Feature Importance')
st.pyplot(fig)

# Second page: Use the model for prediction
st.title('Use the Model for Prediction')

# User input for feature values
st.sidebar.markdown('<h2 style="color: blue;">Choose the values of the following inputs to predict the output</h2>', unsafe_allow_html=True)
user_input_prediction = {}
for column in input_variables_model:
    user_input_prediction[column] = st.sidebar.slider(f'Select {column}', float(data[column].min()), float(data[column].max()), float(data[column].mean()))

# Predict and display result
prediction = model.predict(pd.DataFrame([user_input_prediction]))
st.subheader('Prediction')
st.write(f'The predicted {output_variable_model} value is: {prediction[0]:.5f}')

# Display a bar chart for the predicted output
st.subheader('Predicted Output Chart')
prediction_data = pd.DataFrame({output_variable_model: [prediction[0]]})
fig, ax=plt.subplots(figsize=(8, 5))
sns.barplot(data=prediction_data, palette=['orange'])
ax.set_title(f'Predicted {output_variable_model} Value')
ax.set_ylabel('Value')
st.pyplot(fig)


