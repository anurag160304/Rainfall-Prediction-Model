import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Rainfall Prediction App ☁️🌧️",
    page_icon="🌦️",
    layout="centered"
)

# Load the model and pre-processing tools
try:
    model_xgb = joblib.load('model_xgb.pkl')
    scaler = joblib.load('scaler.pkl')
    lencoders = joblib.load('lencoders.pkl')
    
except:
    st.sidebar.error("❌ Missing model or encoder files.")
    st.stop()
st.sidebar.success("  Made By Baba Baghel!")
# Feature list (Date removed)
features_list = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
                 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']

# Label classes
locations = lencoders['Location'].classes_
wind_gust_dirs = lencoders['WindGustDir'].classes_
wind_dir9am = lencoders['WindDir9am'].classes_
wind_dir3pm = lencoders['WindDir3pm'].classes_

# App Header
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>🌧️ Rainfall Prediction Web App ☀️</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center; color: grey;'>Enter the weather conditions to know the chances of rain tomorrow.</h5>",
    unsafe_allow_html=True
)

# Input Data Dictionary
input_data = {}

# Input Form (Using Expanders)
with st.expander("📍 Location & Rain Info"):
    input_data['Location'] = st.selectbox("Select Location", locations)
    input_data['RainToday'] = st.selectbox("Did it rain today?", ('No', 'Yes'))

with st.expander("🌡️ Temperature & Pressure"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['MinTemp'] = st.number_input("Min Temp (°C)", value=10.0)
        input_data['Temp9am'] = st.number_input("Temp at 9am (°C)", value=15.0)
        input_data['Pressure9am'] = st.number_input("Pressure at 9am (hPa)", value=1010.0)
    with col2:
        input_data['MaxTemp'] = st.number_input("Max Temp (°C)", value=25.0)
        input_data['Temp3pm'] = st.number_input("Temp at 3pm (°C)", value=20.0)
        input_data['Pressure3pm'] = st.number_input("Pressure at 3pm (hPa)", value=1010.0)

with st.expander("☁️ Rainfall, Sunshine & Evaporation"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['Rainfall'] = st.number_input("Rainfall (mm)", value=0.0)
        input_data['Evaporation'] = st.number_input("Evaporation (mm)", value=5.0)
    with col2:
        input_data['Sunshine'] = st.number_input("Sunshine (hrs)", value=8.0)

with st.expander("💨 Wind Details"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['WindGustDir'] = st.selectbox("Wind Gust Direction", wind_gust_dirs)
        input_data['WindDir9am'] = st.selectbox("Wind Dir at 9am", wind_dir9am)
        input_data['WindSpeed9am'] = st.number_input("Wind Speed 9am (km/h)", value=10.0)
    with col2:
        input_data['WindDir3pm'] = st.selectbox("Wind Dir at 3pm", wind_dir3pm)
        input_data['WindGustSpeed'] = st.number_input("Wind Gust Speed (km/h)", value=20.0)
        input_data['WindSpeed3pm'] = st.number_input("Wind Speed 3pm (km/h)", value=15.0)

with st.expander("💧 Humidity & Clouds"):
    col1, col2 = st.columns(2)
    with col1:
        input_data['Humidity9am'] = st.number_input("Humidity at 9am (%)", value=60.0)
        input_data['Cloud9am'] = st.number_input("Cloud at 9am (oktas)", value=4.0)
    with col2:
        input_data['Humidity3pm'] = st.number_input("Humidity at 3pm (%)", value=60.0)
        input_data['Cloud3pm'] = st.number_input("Cloud at 3pm (oktas)", value=4.0)

# Prediction
if st.button("🔮 Predict Rain Tomorrow"):
    input_df = pd.DataFrame([input_data])

    # Label Encoding
    for col in ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if input_df[col][0] in lencoders[col].classes_:
            input_df[col] = lencoders[col].transform(input_df[col])
        else:
            input_df[col] = lencoders[col].transform([lencoders[col].classes_[0]])

    # Encode RainToday
    input_df['RainToday'] = input_df['RainToday'].replace({'No': 0, 'Yes': 1})

    # Reorder & Scale
    input_df = input_df[features_list]
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model_xgb.predict(scaled_input)[0]
    probability = model_xgb.predict_proba(scaled_input)[0][1]

    # Display Result
    st.markdown("---")
    st.subheader("🌤️ Prediction Result:")
    if prediction == 1:
        st.success("🌧️ **It is likely to rain tomorrow!** ☔")
    else:
        st.info("☀️ **It is unlikely to rain tomorrow.**")
    st.write(f"**Probability of Rain Tomorrow:** `{probability:.2%}`")
