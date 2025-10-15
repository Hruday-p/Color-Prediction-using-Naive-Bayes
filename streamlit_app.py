import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import joblib
import numpy as np
import colour
from matplotlib.colors import to_rgb, CSS4_COLORS
import matplotlib.pyplot as plt
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Live pH & Color Predictor",
    page_icon="🧪",
    layout="wide"
)

# --- FIREBASE INITIALIZATION ---
@st.cache_resource
def init_firebase():
    """Initialize the Firebase app, returns a reference to the database."""
    try:
        creds_dict = {
            "type": st.secrets["firebase_credentials"]["type"],
            "project_id": st.secrets["firebase_credentials"]["project_id"],
            "private_key_id": st.secrets["firebase_credentials"]["private_key_id"],
            "private_key": st.secrets["firebase_credentials"]["private_key"],
            "client_email": st.secrets["firebase_credentials"]["client_email"],
            "client_id": st.secrets["firebase_credentials"]["client_id"],
            "auth_uri": st.secrets["firebase_credentials"]["auth_uri"],
            "token_uri": st.secrets["firebase_credentials"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase_credentials"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase_credentials"]["client_x509_cert_url"],
        }
        database_url = st.secrets["firebase_database"]["databaseURL"]
        
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        
        return db.reference('/')
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None

db_ref = init_firebase()

# --- LOAD ML MODEL AND SCALER ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("gaussian_nb.pkl")
        scaler = joblib.load("scaler_nb.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler files: {e}")
        return None, None

ml_model, scaler = load_model_and_scaler()

# --- HELPER FUNCTIONS ---
def rgb_to_xy(rgb):
    xyz = colour.sRGB_to_XYZ(np.array(rgb) / 255.0)
    return colour.XYZ_to_xy(xyz)

def closest_color_name(rgb):
    min_dist, closest = float('inf'), "Unknown"
    for name, hex_val in CSS4_COLORS.items():
        dist = np.linalg.norm(np.array(to_rgb(hex_val)) * 255 - np.array(rgb))
        if dist < min_dist:
            min_dist, closest = dist, name
    return closest

# --- UI LAYOUT ---
st.title("🧪 Live pH & Color Predictor")
st.markdown("This application controls a remote color sensor and predicts the pH category of a sample in real-time.")

if db_ref is None or ml_model is None:
    st.warning("Application cannot start. Check Firebase connection and model files.")
else:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Hardware Control")
        if st.button("🟢 Sensor LED ON"):
            db_ref.child('control').set({'command': 'sensor_on'})
        if st.button("🔴 Sensor LED OFF"):
            db_ref.child('control').set({'command': 'sensor_off'})
        if st.button("🟣 UV LED ON"):
            db_ref.child('control').set({'command': 'uv_on'})
        if st.button("⚫ UV LED OFF"):
            db_ref.child('control').set({'command': 'uv_off'})

        st.subheader("Live Data")
        data_placeholder = st.empty()

    with col2:
        st.subheader("Live Chromaticity Diagram")
        plot_placeholder = st.empty()

    # --- MAIN LIVE LOOP ---
    while True:
        try:
            sensor_data = db_ref.child('sensor_data').get()

            if sensor_data and 'r' in sensor_data:
                r, g, b = sensor_data['r'], sensor_data['g'], sensor_data['b']
                rgb = [r, g, b]
                
                rgb_scaled = scaler.transform(np.array([rgb]))
                prediction = ml_model.predict(rgb_scaled)[0]
                color_name = closest_color_name(rgb)
                
                with data_placeholder.container():
                    st.color_picker("Current Color", f"rgb({r},{g},{b})", key="color_picker")
                    st.metric("ML Predicted pH Category", prediction)
                    st.metric("Closest Color Name", color_name)
                    st.text(f"Raw RGB: ({r}, {g}, {b})")

                with plot_placeholder.container():
                    fig, ax = plt.subplots()
                    wavelength = np.arange(380, 780, 5)
                    xy_gamut = colour.XYZ_to_xy(colour.wavelength_to_XYZ(wavelength))
                    ax.plot(xy_gamut[:, 0], xy_gamut[:, 1], color="black", linewidth=1)
                    
                    xy_val = rgb_to_xy(rgb)
                    color_norm = [v / 255.0 for v in rgb]
                    
                    ax.plot(xy_val[0], xy_val[1], "o", markersize=15, color=color_norm, markeredgecolor='black')
                    ax.set_title("CIE 1931 Chromaticity Diagram")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_xlim(0, 0.8)
                    ax.set_ylim(0, 0.9)
                    ax.grid(True)
                    st.pyplot(fig)
            
            else:
                 with data_placeholder.container():
                    st.info("Waiting for sensor data... Turn on the Sensor LED to begin.")

            time.sleep(1)

        except Exception as e:
            st.error(f"An error occurred during the live update: {e}")
            time.sleep(5)
