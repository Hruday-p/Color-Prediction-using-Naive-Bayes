import streamlit as st
import pandas as pd
import numpy as np
import joblib
import colour
from matplotlib.colors import to_rgb, CSS4_COLORS
import matplotlib.pyplot as plt
import os
import time
import firebase_admin
from firebase_admin import credentials, db

# ==============================================================================
# --- Page Configuration & Title ---
# ==============================================================================
st.set_page_config(
    page_title="Live pH & Color Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Live pH & Color Predictor")
# --- TEMPORARY DEBUGGING CODE ---
# Add this to see exactly what Streamlit is reading from your secrets.
st.subheader("⚠️ DEBUGGING INFORMATION")
try:
    creds_to_check = st.secrets["firebase_credentials"]
    st.write("Type of `firebase_credentials` secret:", type(creds_to_check))
    # The line below will display the credentials in a structured way.
    st.json(creds_to_check)
except Exception as e:
    st.error(f"Could not access secrets['firebase_credentials']: {e}")
# --- END OF DEBUGGING CODE ---
st.write("This web app uses a trained Machine Learning model to predict the pH category based on live RGB color values from a sensor.")

# ==============================================================================
# --- Firebase Initialization ---
# ==============================================================================

@st.cache_resource
def init_firebase():
    """Initialize the Firebase app, returns a reference to the database."""
    try:
        # Check if the app is already initialized
        if not firebase_admin._apps:
            # Load credentials from Streamlit's secrets
            cred_dict = st.secrets["firebase_credentials"]
            database_url = st.secrets["firebase_database"]["databaseURL"]
            
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': database_url
            })
        
        # Return a reference to the root of the database
        return db.reference('/')
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.info("Please ensure your Firebase credentials and database URL are correctly configured in Streamlit's secrets.")
        return None

db_ref = init_firebase()

# ==============================================================================
# --- Session State Initialization ---
# ==============================================================================
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'latest_rgb' not in st.session_state:
    st.session_state.latest_rgb = [220, 220, 220] 

# ==============================================================================
# --- Load Model and Helper Functions ---
# ==============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained model and scaler."""
    try:
        model = joblib.load("gaussian_nb.pkl")
        scaler = joblib.load("scaler_nb.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: 'gaussian_nb.pkl' or 'scaler_nb.pkl' not found.")
        return None, None

ml_model, scaler = load_model_and_scaler()

# === Helper Functions ===
ph_reference = {
    1: (255, 0, 0), 2: (255, 51, 0), 3: (255, 102, 0), 4: (255, 153, 0),
    5: (255, 204, 0), 6: (204, 255, 0), 7: (0, 255, 0), 8: (0, 255, 128),
    9: (0, 204, 255), 10: (0, 102, 255), 11: (102, 0, 255), 12: (153, 0, 255),
    13: (204, 0, 255), 14: (153, 0, 153)
}

def rgb_to_xy(rgb):
    rgb_normalized = np.array(rgb) / 255.0
    if np.all(rgb_normalized == 0): return [0.3127, 0.3290]
    xyz = colour.sRGB_to_XYZ(rgb_normalized)
    return colour.XYZ_to_xy(xyz)

def closest_color_name(rgb):
    min_dist, closest = float('inf'), "Unknown"
    for name, hex_val in CSS4_COLORS.items():
        dist = np.linalg.norm(np.array(to_rgb(hex_val)) * 255 - np.array(rgb))
        if dist < min_dist: min_dist, closest = dist, name
    return closest

def estimate_ph_from_ref(rgb):
    min_dist, est_ph = float('inf'), None
    for ph, ref_rgb in ph_reference.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(ref_rgb))
        if dist < min_dist: min_dist, est_ph = dist, ph
    return est_ph

# ==============================================================================
# --- User Interface & Prediction Logic ---
# ==============================================================================

if ml_model is None or scaler is None or db_ref is None:
    st.stop()

# --- Control Buttons and Status Display ---
st.header("Controls")
col_control1, col_control2, col_status = st.columns([1, 1, 3])

with col_control1:
    if st.button("▶️ Start Reading", use_container_width=True):
        st.session_state.is_running = True
        # Send 'start' command to Firebase
        db_ref.child("control").set({"command": "start", "timestamp": time.time()})
        st.rerun()

with col_control2:
    if st.button("⏹️ Stop Reading", use_container_width=True):
        st.session_state.is_running = False
        # Send 'stop' command to Firebase
        db_ref.child("control").set({"command": "stop", "timestamp": time.time()})
        st.rerun()

with col_status:
    if st.session_state.is_running:
        st.success("Status: Reading live data... (Awaiting data from local script)")
    else:
        st.warning("Status: Stopped. Click 'Start Reading' to begin.")

st.divider()

# --- Data Fetching and Display ---
if st.session_state.is_running:
    # Fetch the latest RGB data from Firebase
    firebase_data = db_ref.child("data").get()
    if firebase_data and 'rgb' in firebase_data and isinstance(firebase_data['rgb'], list) and len(firebase_data['rgb']) == 3:
        st.session_state.latest_rgb = firebase_data['rgb']
    
current_rgb = st.session_state.latest_rgb
col1, col2 = st.columns([1, 2])

# UI rendering remains largely the same, using `current_rgb`
with col1:
    st.subheader("Current Color")
    color_norm = [v / 255.0 for v in current_rgb]
    fig_color, ax_color = plt.subplots(figsize=(2, 2)); ax_color.imshow([[color_norm]]); ax_color.axis('off'); st.pyplot(fig_color)
    st.subheader("Live RGB Values")
    st.write(f"**R:** {current_rgb[0]}, **G:** {current_rgb[1]}, **B:** {current_rgb[2]}")

with col2:
    st.subheader("Analysis & Predictions")
    if sum(current_rgb) == 0:
        ml_ph, est_ph, color_name = "Black", "-", "Black"
    else:
        scaled_rgb = scaler.transform([current_rgb])
        ml_ph = ml_model.predict(scaled_rgb)[0]
        est_ph = estimate_ph_from_ref(current_rgb)
        color_name = closest_color_name(current_rgb)
    st.metric("🤖 ML Predicted pH Category", str(ml_ph))
    st.metric("🎨 Closest Color Name", color_name)
    st.metric("📊 Estimated pH (from Ref. Table)", str(est_ph))

# --- Chromaticity Diagram ---
st.subheader("CIE 1931 Chromaticity Diagram")
fig_cie, ax_cie = plt.subplots(figsize=(8, 6))
wavelength = np.arange(380, 780, 5)
xy_gamut = colour.XYZ_to_xy(colour.wavelength_to_XYZ(wavelength))
ax_cie.plot(xy_gamut[:, 0], xy_gamut[:, 1], "k-"); ax_cie.set_title("Color Location in CIE 1931 Space")
ax_cie.set_xlabel("x"); ax_cie.set_ylabel("y"); ax_cie.set_xlim(0, 0.8); ax_cie.set_ylim(0, 0.9); ax_cie.grid(True)
xy_val = rgb_to_xy(current_rgb)
ax_cie.plot(xy_val[0], xy_val[1], "o", markersize=12, markeredgecolor='k', color=[v/255.0 for v in current_rgb]); st.pyplot(fig_cie)

# --- Cloud Deployment Notes ---
with st.expander("ℹ️ How This Works", expanded=True):
    st.info("""
        - When you click **Start/Stop**, this app writes a command to a Firebase Realtime Database.
        - A **local Python script** (running on the computer with the Arduino) listens for these commands.
        - The local script turns the sensor ON/OFF and sends the live RGB data back to the Firebase database.
        - This web app reads that data from Firebase and updates the display in real-time.
    """)

# --- Auto-refresh Loop ---
if st.session_state.is_running:
    time.sleep(1) # Refresh every 1 second
    st.rerun()