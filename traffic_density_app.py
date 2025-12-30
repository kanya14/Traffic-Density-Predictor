import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# --- Load Assets ---
base_path = os.path.dirname(__file__)
model = tf.keras.models.load_model(os.path.join(base_path, "traffic_model.h5"), compile=False)

with open(os.path.join(base_path, "label_encoders.pkl"), 'rb') as f:
    encoders = pickle.load(f)
with open(os.path.join(base_path, "cols_list.pkl"), 'rb') as f:
    config = pickle.load(f)

# --- App Header ---
st.title("🚦 Urban Traffic Density Predictor")
st.markdown("""
### Smart AI for Future City Planning
This application uses a **Machine Learning model** to estimate how crowded city roads will be. 
By analyzing factors like weather, vehicle types, and economic status, it predicts a 'Density Score' to help you understand traffic flow.
""")
st.divider()

# --- Input Section ---
st.header("Step 1: Environment & City Details")
inputs = {}

# Make categorical inputs more friendly
col_map = {
    "City": "Which city are you checking?",
    "Vehicle_Type": "Primary type of vehicle on the road?",
    "Weather": "What is the current weather like?",
    "Economic_Condition": "What is the current economic state?",
    "Day_Of_Week": "What day is it today?"
}

for col in config['cat_cols']:
    label = col_map.get(col, f"Select {col}")
    options = encoders[col].classes_.tolist()
    selection = st.selectbox(label, options)
    inputs[f"{col}_input"] = np.array([encoders[col].transform([selection])[0]])

st.divider()
st.header("Step 2: Real-Time Traffic Conditions")

# Create two columns for a cleaner look
left_col, right_col = st.columns(2)

with left_col:
    hour = st.slider("Time of Day (24h Format)", 0, 23, 12, 
                     help="Select the hour from 0 (Midnight) to 23 (11 PM).")
    speed = st.number_input("Average Traffic Speed", value=17.40, 
                            help="The average speed of vehicles currently on the road.")
    energy = st.number_input("Energy Consumption", value=10.40, 
                             help="Measured energy usage of the transport network.")

with right_col:
    is_peak_str = st.selectbox("Is it currently Rush Hour?", ["No", "Yes"], 
                               help="Select 'Yes' if this is typical peak morning or evening commute time.")
    event_str = st.selectbox("Is there a special event? (e.g., accidents, road closures)", ["No", "Yes"], 
                             help="Select 'Yes' if there is an accident, road closure, or major public event.")

# Mapping Yes/No to 0/1 for the model
mapping = {"No": 0, "Yes": 1}
is_peak = mapping[is_peak_str]
event = mapping[event_str]

# --- Prediction Section ---
st.divider()
if st.button("🔍 Calculate Traffic Density", use_container_width=True):
    with st.spinner('Analyzing city data...'):
        # Prepare numeric array
        num_array = np.array([[hour, speed, is_peak, event, energy]])
        inputs["numeric_input"] = num_array
        
        # Inference
        prediction = model.predict(inputs)
        result = prediction[0][0]
        
        # Display Result
        st.subheader("Results")
        st.metric(label="Predicted Traffic Density Score", value=f"{result:.4f}")
        
        if result > 0.7:
            st.error("Warning: High traffic density expected. Expect heavy delays!")
        elif result > 0.4:
            st.warning("Moderate traffic density predicted. Plan accordingly.")
        else:
            st.success("Low traffic density predicted. Roads should be clear!")