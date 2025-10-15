import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import colour
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Live pH & Color Predictor",
    page_icon="🧪",
    layout="wide",
)

# --- FIREBASE INITIALIZATION ---
# This is cached for performance, so it only runs once.
@st.cache_resource
def init_firebase():
    """Initialize the Firebase app, returns a reference to the database."""
    try:
        if not firebase_admin._apps:
            # Build the credentials dictionary from secrets
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
        
        return db.reference('/')
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.warning("Please ensure your Firebase credentials and database URL are correctly configured in Streamlit's secrets.")
        return None

# --- LOAD ML MODEL AND SCALER ---
@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    try:
        model = joblib.load('gaussian_nb.pkl')
        scaler = joblib.load('scaler_nb.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please ensure 'gaussian_nb.pkl' and 'scaler_nb.pkl' are in the repository.")
        return None, None

ml_model, scaler = load_model()

# --- HELPER FUNCTIONS ---
def rgb_to_xy(rgb):
    """Convert RGB to CIE xy chromaticity coordinates."""
    try:
        xyz = colour.sRGB_to_XYZ(np.array(rgb) / 255.0)
        return colour.XYZ_to_xy(xyz)
    except Exception:
        return (0.33, 0.33) # Default to white point if conversion fails

# --- MAIN APP ---
db_ref = init_firebase()

if db_ref and ml_model and scaler:
    st.title("🧪 Live pH & Color Predictor")
    
    # --- UI LAYOUT ---
    control_col, display_col = st.columns([1, 2])

    with control_col:
        st.subheader("💡 LED Controls")
        
        # --- NEW CONTROL BUTTONS ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sensor LED ON", use_container_width=True):
                db_ref.child('control').set({'command': 'sensor_on'})
                st.toast("Sent 'Sensor ON' command")
            if st.button("UV LED ON", use_container_width=True):
                db_ref.child('control').set({'command': 'uv_on'})
                st.toast("Sent 'UV ON' command")
        with col2:
            if st.button("Sensor LED OFF", use_container_width=True):
                db_ref.child('control').set({'command': 'sensor_off'})
                st.toast("Sent 'Sensor OFF' command")
            if st.button("UV LED OFF", use_container_width=True):
                db_ref.child('control').set({'command': 'uv_off'})
                st.toast("Sent 'UV OFF' command")
        
        st.divider()
        
        st.subheader("📊 Live Data")
        status_placeholder = st.empty()
        color_placeholder = st.empty()
        prediction_placeholder = st.empty()

    with display_col:
        st.subheader("CIE 1931 Chromaticity Diagram")
        chart_placeholder = st.empty()

    # --- LIVE DATA LISTENER ---
    # This uses a generator to continuously listen for changes in Firebase.
    def data_listener():
        try:
            for event in db_ref.child('sensor_data').listen():
                if event.data and isinstance(event.data, dict):
                    yield event.data
        except Exception:
            yield None

    for live_data in data_listener():
        if live_data:
            r, g, b = live_data.get('r', 0), live_data.get('g', 0), live_data.get('b', 0)
            rgb_color = [r, g, b]
            
            # --- UPDATE UI ELEMENTS ---
            with status_placeholder.container():
                st.info("Receiving live data...")
            
            with color_placeholder.container():
                st.markdown(f"**RGB:** `{r}, {g}, {b}`")
                st.color_picker("Live Color", value=f"#{r:02x}{g:02x}{b:02x}", disabled=True, label_visibility="collapsed")

            # --- ML Prediction ---
            with prediction_placeholder.container():
                try:
                    scaled_data = scaler.transform([rgb_color])
                    prediction = ml_model.predict(scaled_data)[0]
                    st.metric("Predicted pH Category", prediction)
                except Exception as e:
                    st.warning(f"Could not make prediction: {e}")
            
            # --- UPDATE CHART ---
            with chart_placeholder.container():
                xy_val = rgb_to_xy(rgb_color)
                chart_data = pd.DataFrame({'x': [xy_val[0]], 'y': [xy_val[1]]})
                st.scatter_chart(
                    chart_data, x='x', y='y',
                    color=[f"#{r:02x}{g:02x}{b:02x}"],
                    size=200
                )
        else:
            with status_placeholder.container():
                st.warning("Waiting for data... Turn on Sensor LED to start.")

