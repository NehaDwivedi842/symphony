import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import streamlit as st

# Load the data for fitting the scaler
data = pd.read_csv('Load_Calculations_Training_Dataset.csv')  # Replace with your data file
data = data.drop(columns=['Unnamed: 0'])

# Separate input features (X) and output values (y)
data.dropna(inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

# Load the MinMaxScaler and fit it to the subset of data
scaler = MinMaxScaler()
scaler.fit(X)

# Load the trained model
model = load_model('trained_model.h5')  # Replace with the path to your trained model
model.compile(loss='mse', optimizer='adam')  # Compile the model with Mean Squared Error loss


# Streamlit App
st.set_page_config(
    page_title="Tonnage Prediction App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)


# Center the title
st.markdown("""
<style>
h1 {
  text-align: center;
}
button {
  margin: 0 auto;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #337ab7;
  color: white;
  font-size: 16px;
  border-radius: 5px;
  padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

st.title(":chart_with_upwards_trend: Tonnage Prediction")
st.markdown("---")  # Horizontal line for separation

# Add input fields for user to enter data
st.header("Enter Input Data")

# Create column layout
col1, col2 = st.columns(2)

with col1:
    m = st.number_input("Projected Area (sq.in)", step=0.0001)
    h = st.number_input("Flow Length (in)", step=0.0001)

with col2:
    c = st.number_input("Thickness (in)", step=0.0001)
    p = st.number_input("Peak Injection Pressure (psi)", step=0.0001)

t = st.number_input("Pack Pressures Used (psi)", step=0.0001)

# Add a submit button with custom style
st.markdown("---")  # Horizontal line for separation
submitted = st.button("Submit", key="prediction_button")

if submitted:
    # Prepare and normalize new data
    new_data = [[m, c, h, p, t]]
    new_data_scaled = scaler.transform(new_data)

    # Make predictions using the model
    predicted_values = model.predict(new_data_scaled)

    # Predict load
    predicted_load = predicted_values[0][0]  # Access the prediction value

    # Display the predicted load with a highlighted result
    st.success(f"Predicted Load: {predicted_load:.2f} tons")





