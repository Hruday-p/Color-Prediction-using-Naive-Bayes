import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import joblib
import colour
from matplotlib.colors import to_rgb, CSS4_COLORS
import matplotlib.pyplot as plt
import time

# --- App Functions ---
@st.cache_resource
def init_firebase():
    """Initialize the Firebase app, returns a reference to the database."""
    try:
        # Check if the app is already initialized
        if not firebase_admin._apps:
            # Manually build a standard Python dictionary from secrets
            # This is the most robust method to ensure compatibility.
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
            
            cred = credentials.Certificate(creds_dict)
            
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })
        
        # Return a reference to the root of the database
        return db.reference('/')
    except Exception as e:
        # This will now catch the error more gracefully
        st.error(f"Failed to initialize Firebase: {e}")
        st.info("Please ensure your Firebase credentials and database URL are correctly configured in Streamlit's secrets.")
        return None

@st.cache_data
def load_model_and_scaler():
    """Loads the ML model and scaler."""
    model = joblib.load("gaussian_nb.pkl")
    scaler = joblib.load("scaler_nb.pkl")
    return model, scaler

def rgb_to_xy(rgb):
    """Converts RGB to CIE 1931 xy coordinates."""
    # Normalize RGB values
    rgb_normalized = np.array(rgb) / 255.0
    # Convert sRGB to XYZ
    xyz = colour.sRGB_to_XYZ(rgb_normalized)
    # Convert XYZ to xy
    return colour.XYZ_to_xy(xyz)

def closest_color_name(rgb):
    """Finds the closest CSS4 color name for a given RGB value."""
    min_dist = float('inf')
    closest_name = "Unknown"
    for name, hex_val in CSS4_COLORS.items():
        ref_rgb = np.array(to_rgb(hex_val)) * 255
        dist = np.linalg.norm(np.array(rgb) - ref_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

# --- Main App Logic ---
st.set_page_config(layout="wide")
st.title("🔬 Real-Time pH Predictor Dashboard")
st.markdown("This dashboard displays live color data from a sensor and predicts the pH category using a trained machine learning model.")

# Initialize Firebase and load models
db_ref = init_firebase()
ml_model, scaler = load_model_and_scaler()

if not db_ref:
    st.stop()

# --- UI Layout ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("🎮 Sensor Control & Status")
    
    # Control Buttons
    if st.button("▶️ Start Reading", key="start"):
        db_ref.child('control').set({'command': 'start'})
        st.success("Sent 'Start' command!")

    if st.button("⏹️ Stop Reading", key="stop"):
        db_ref.child('control').set({'command': 'stop'})
        st.warning("Sent 'Stop' command!")

    st.markdown("---")
    
    # Live Data Display
    status_placeholder = st.empty()
    color_placeholder = st.empty()
    metrics_placeholder = st.empty()

with col2:
    st.header("📊 Chromaticity Diagram (Live)")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the spectral locus
    wavelengths = np.arange(380, 781, 5)
    xy_locus = colour.XYZ_to_xy(colour.wavelength_to_XYZ(wavelengths))
    ax.plot(xy_locus[:, 0], xy_locus[:, 1], color='black', linewidth=1)
    
    ax.set_title("CIE 1931 Chromaticity Diagram")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    # Live data point
    live_point, = ax.plot([], [], 'o', markersize=15, markeredgecolor='black')
    
    plot_placeholder = st.pyplot(fig)


# --- Real-Time Loop ---
while True:
    try:
        # Listen for data changes from Firebase
        data = db_ref.child('sensor_data').get()
        status = db_ref.child('control').get()

        if data and isinstance(data, dict):
            rgb = [data.get('r', 0), data.get('g', 0), data.get('b', 0)]
            
            # --- Update UI Elements ---
            with status_placeholder.container():
                 st.subheader("Live Status")
                 current_status = status.get('command', 'stopped').capitalize() if status else 'Stopped'
                 st.metric(label="Sensor Status", value=current_status)

            with color_placeholder.container():
                st.subheader("Live Color Preview")
                st.color_picker("Current color from sensor:", f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})", disabled=True)
            
            # Perform predictions
            scaled_rgb = scaler.transform([rgb])
            prediction = ml_model.predict(scaled_rgb)[0]
            color_name = closest_color_name(rgb)

            with metrics_placeholder.container():
                st.subheader("Live Predictions & Metrics")
                st.metric(label="ML Predicted pH Category", value=str(prediction))
                st.metric(label="Closest Color Name", value=color_name)
                st.text(f"Raw RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}")

            # Update plot
            xy_val = rgb_to_xy(rgb)
            live_point.set_data([xy_val[0]], [xy_val[1]])
            live_point.set_color(np.array(rgb) / 255.0)
            plot_placeholder.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during the live update loop: {e}")
        break # Exit the loop on error to prevent crashing
    
    time.sleep(1) # Refresh every second

