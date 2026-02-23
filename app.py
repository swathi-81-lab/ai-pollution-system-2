import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
with open('pollution_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app title
st.title("AI-Based Pollution Monitoring System 🌍")

st.write("Enter environmental and traffic data to predict PM2.5 levels:")

# User input
temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=3.0)
wind_direction = st.number_input("Wind Direction (°)", min_value=0, max_value=360, value=180)
PM10 = st.number_input("PM10 (µg/m³)", min_value=0, max_value=500, value=100)
NO2 = st.number_input("NO2 (ppb)", min_value=0, max_value=200, value=40)
SO2 = st.number_input("SO2 (ppb)", min_value=0, max_value=100, value=10)
CO = st.number_input("CO (ppm)", min_value=0.0, max_value=10.0, value=0.8)
O3 = st.number_input("O3 (ppb)", min_value=0, max_value=200, value=30)
traffic_volume = st.number_input("Traffic Volume (vehicles/hour)", min_value=0, max_value=10000, value=200)

# Predict button
if st.button("Predict PM2.5"):
    input_data = np.array([[PM10, NO2, SO2, CO, O3, temperature, humidity, wind_speed, wind_direction, traffic_volume]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted PM2.5 level: {prediction[0]:.2f} µg/m³")