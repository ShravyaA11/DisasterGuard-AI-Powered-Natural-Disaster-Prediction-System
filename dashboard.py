# dashboard.py
# Streamlit dashboard for analytics + prediction
# Project structure:
#  - data/Natural disaster.csv or cleaned_disaster.csv (optional for analytics)
#  - src/final_disaster_model.pkl (trained model)
#  - src/label_encoder.pkl (optional for decoding predictions)
#  - src/selected_features.pkl (optional feature list)

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from pathlib import Path
import plotly.express as px

st.set_page_config(layout='wide', page_title='DisasterGuard Dashboard')

BASE = Path('.')
DATA_DIR = BASE / 'data'
SRC_DIR = BASE / 'src'

CANDIDATE_DATA = [DATA_DIR / 'cleaned_disaster.csv', DATA_DIR / 'Natural disaster.csv']
MODEL_FILE = SRC_DIR / 'final_disaster_model.pkl'
LABEL_ENCODER_FILE = SRC_DIR / 'label_encoder.pkl'
FEATURES_FILE = SRC_DIR / 'selected_features.pkl'

@st.cache_data
def load_dataset():
    for f in CANDIDATE_DATA:
        if f.exists():
            return pd.read_csv(f)
    return None

@st.cache_resource
def load_model():
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            with open(MODEL_FILE, 'rb') as f:
                return pickle.load(f)
    return None

@st.cache_resource
def load_pickle(p):
    if p.exists():
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(p)
    return None

# Load resources
df = load_dataset()
model = load_model()
label_encoder = load_pickle(LABEL_ENCODER_FILE)
selected_features = load_pickle(FEATURES_FILE)

st.title('🌍 DisasterGuard — Analytics & Prediction')

# ------------------- Analytics -------------------
st.header('Analytics')
if df is not None:
    target_col = None
    for c in ['target','label','disaster','class','Category','is_disaster']:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]

    st.subheader('Disaster type distribution')
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'count']
    fig = px.bar(counts, x=target_col, y='count', color=target_col,
                 title='Occurrences of each disaster type', text='count')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Yearly trend (if year/time column exists)')
    time_cols = [c for c in df.columns if 'year' in c.lower() or 'date' in c.lower() or 'time' in c.lower()]
    if time_cols:
        col = time_cols[0]
        try:
            df['_t'] = pd.to_datetime(df[col])
            trend = df.groupby(df['_t'].dt.to_period('Y'))[target_col].value_counts().reset_index(name='count')
            trend['_t'] = trend['_t'].astype(str)
            fig2 = px.bar(trend, x='_t', y='count', color=target_col, barmode='group',
                          title=f'Disaster counts over time ({col})')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.info(f'Time trend could not be created: {e}')
    else:
        st.info('No time column detected for trend analysis.')
else:
    st.info('No dataset found in data/. Only prediction will be available.')

# ------------------- Prediction -------------------
# ------------------- Prediction -------------------
st.header('Prediction')
if model is None:
    st.error('No trained model found at src/final_disaster_model.pkl')
else:
    # Disaster mapping (from first code)
    disaster_mapping = {
        0: "Flood",
        1: "Heatwave",
        2: "Earthquake",
        3: "Cyclone",
        4: "Drought",
        5: "Landslide",
        6: "Wildfire",
        7: "Tsunami",
        8: "Volcanic Eruption"
    }

    # Sidebar for inputs
    st.sidebar.header("Enter Disaster Data")
    year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, value=2021)
    deaths = st.sidebar.number_input("Total Deaths", min_value=0, value=0)
    affected = st.sidebar.number_input("Total Affected", min_value=0, value=0)
    damages = st.sidebar.number_input("Total Damages ('000 US$)", min_value=0, value=0)

    if st.sidebar.button("Predict Disaster"):
        try:
            # --- Build input with ALL features model expects ---
            input_data = {
                'Year': year,
                'Seq': 0,
                'Disaster Group': 'Unknown',
                'Disaster Subgroup': 'Unknown',
                'Disaster Subtype': 'Unknown',
                'Disaster Subsubtype': 'Unknown',
                'Country': 'Unknown',
                'ISO': 'UNK',
                'Region': 'Unknown',
                'Continent': 'Unknown',
                'Location': 'Unknown',
                'Origin': 'Unknown',
                'Associated Dis': 'None',
                'Associated Dis2': 'None',
                'OFDA Response': 'No',
                'Appeal': 'No',
                'Declaration': 'No',
                'Aid Contribution': 0,
                'Dis Mag Value': 0,
                'Dis Mag Scale': 'NA',
                'Latitude': 0.0,
                'Longitude': 0.0,
                'Local Time': '00:00:00',
                'River Basin': 'Unknown',
                'Start Year': year,
                'Start Month': 1,
                'Start Day': 1,
                'End Year': year,
                'End Month': 1,
                'End Day': 1,
                'Total Deaths': deaths,
                'No Injured': 0,
                'No Affected': 0,
                'No Homeless': 0,
                'Total Affected': affected,
                "Insured Damages ('000 US$)": 0,
                "Total Damages ('000 US$)": damages,
                'CPI': 0
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Convert object columns to categorical
            for col in input_df.select_dtypes(include=['object']).columns:
                input_df[col] = input_df[col].astype('category')

            # Reorder columns to match training data
            expected_features = model.get_booster().feature_names
            input_df = input_df[expected_features]

            # Predict (numeric label)
            prediction_num = int(model.predict(input_df, validate_features=False)[0])

            # Map number → disaster name
            prediction_label = disaster_mapping.get(prediction_num, f"Unknown ({prediction_num})")

            # Show result
            st.subheader("Prediction Result")
            st.success(f"🚨 Predicted Disaster Type: *{prediction_label}*")

        except Exception as e:
            st.error(f"Prediction failed: {e}")