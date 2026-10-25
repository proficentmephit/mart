import streamlit as st
import pandas as pd
import joblib

# Set page title and layout
st.set_page_config(page_title="Advertising Sales Predictor", layout="centered")

st.title("‑‑‑ Advertising Sales Predictor ‑‑‑")
st.write("Enter the advertising budgets below to predict estimated Sales.")

# Load the saved model (using joblib)
@st.cache_resource
def load_model():
    # Although named .csv, it contains the serialized model
    return joblib.load('model.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Input fields for budgets
st.subheader("Advertising Budgets")
tv_budget = st.slider("TV Budget ($)", min_value=0.0, max_value=350.0, value=150.0, step=0.1)
radio_budget = st.slider("Radio Budget ($)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
newspaper_budget = st.slider("Newspaper Budget ($)", min_value=0.0, max_value=120.0, value=20.0, step=0.1)

# Construct input dataframe matching model expected features
input_df = pd.DataFrame([[tv_budget, radio_budget, newspaper_budget]], 
                        columns=['TV', 'Radio', 'Newspaper'])

# Make prediction on button press
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Sales: {prediction:.2f} units")
