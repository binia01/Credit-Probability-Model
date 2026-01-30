"""
Streamlit Dashboard for Credit Risk Probability Model
Bati Bank Credit Scoring System
"""

from PIL import Image
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import json

# Page configuration
icon = Image.open("assets/Bati.png")
st.set_page_config(
    page_title="Bati Bank - Credit Risk Scoring", page_icon=icon, layout="wide"
)

# Styling
st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; text-align: center; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==================== API Configuration ====================
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_api_predict(customer_id: str, features: dict, is_raw: bool = True) -> dict:
    """Call the FastAPI endpoint for credit scoring."""
    try:
        url = f"{API_BASE_URL}/predict"
        params = {"customer_id": customer_id, "is_raw": is_raw}
        response = requests.post(url, json=features, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Ensure the FastAPI server is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None
