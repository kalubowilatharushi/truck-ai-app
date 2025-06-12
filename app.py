import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import re

# --- 1. PAGE CONFIG and CSS (Copy from Step 1 above) ---
st.set_page_config(page_title="AI Truck Health Prediction", page_icon="🚚", layout="wide")

css_string = """
<style>
    /* Main app background */
    .stApp { background-color: #F0F2F6; }
    /* Card-like containers */
    div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 25px;
        background-color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .main .block-container { padding-top: 2rem; }
    .title { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; padding-bottom: 1rem; }
    h3 { padding-top: 1rem; }
</style>
"""
st.markdown(css_string, unsafe_allow_html=True)


# --- 2. YOUR EXISTING FUNCTIONS AND DATA LOADING ---
# Place your data loading, model prediction, and NLP functions here.
# For demonstration, I'll create dummy functions.

@st.cache_data
def load_data():
    # Replace this with your actual Google Sheets loading logic
    data = {
        'Truck ID': [101, 102, 103, 104],
        'Odometer (km)': [150000, 250000, 80000, 320000],
        'Last Mechanic Notes': [
            'Routine check, oil change, brakes seem fine',
            'Engine making a strange noise, possible injector issue. Replaced filter.',
            'New tires fitted, coolant levels topped up.',
            'Transmission fluid leak detected and fixed. Engine sounds rough on startup.'
        ]
    }
    return pd.DataFrame(data)

def predict_risk(truck_id, odometer, notes):
    # --- INSERT YOUR ACTUAL AI MODEL LOGIC HERE ---
    # This is a dummy logic based on odometer reading.
    if odometer > 300000 or "leak" in notes.lower() or "rough" in notes.lower():
        return "High", 0.95
    elif odometer > 200000 or "noise" in notes.lower() or "issue" in notes.lower():
        return "Medium", 0.65
    else:
        return "Low", 0.15

def analyze_notes(notes):
    # --- INSERT YOUR ACTUAL NLP KEYWORD EXTRACTION LOGIC HERE ---
    words = re.findall(r'\b\w+\b', notes.lower())
    stop_words = set(['a', 'the', 'is', 'in', 'on', 'and', 'to', 'was', 'it', 'for', 'with', 'check', 'change', 'up'])
    keywords = [word for word in words if word.isalpha() and word not in stop_words]
    return [word for word, count in Counter(keywords).most_common(5)]

# --- 3. APP LAYOUT ---

# Load data
df = load_data()

# --- HEADER ---
st.markdown('<p class="title">🚚 AI Truck Health Dashboard</p>', unsafe_allow_html=True)
st.markdown("Select a truck to view its health prediction and analysis.")

# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("Select Truck")
selected_truck_id = st.sidebar.selectbox("Truck ID", df['Truck ID'])

# Get selected truck data
truck_data = df[df['Truck ID'] == selected_truck_id].iloc[0]

# Allow user to override default values
st.sidebar.subheader("Or Enter Manual Data")
odometer_input = st.sidebar.number_input("Odometer (km)", value=truck_data['Odometer (km)'])
notes_input = st.sidebar.text_area("Mechanic Notes", value=truck_data['Last Mechanic Notes'], height=150)

# --- PREDICTION ---
risk_level, risk_score = predict_risk(selected_truck_id, odometer_input, notes_input)
keywords = analyze_notes(notes_input)

# --- DASHBOARD DISPLAY ---
col1, col2 = st.columns((2, 1.5)) # Create two columns with different widths

with col1:
    with st.container():
        st.subheader(f"Analysis for Truck {selected_truck_id}")
        
        # --- GAUGE CHART FOR RISK ---
        risk_color = "green"
        if risk_level == "Medium": risk_color = "orange"
        if risk_level == "High": risk_color = "red"

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>Failure Risk: {risk_level}</b>", 'font': {'size': 24}},
            delta = {'reference': 40, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ccc",
                'steps': [
                    {'range': [0, 40], 'color': 'lightgreen'},
                    {'range': [40, 70], 'color': 'lightyellow'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container():
        st.subheader("Key Details")
        st.metric(label="Odometer Reading", value=f"{odometer_input:,} km")
        
        st.subheader("Mechanic Notes Analysis")
        if keywords:
            for keyword in keywords:
                st.markdown(f"- **{keyword.capitalize()}**")
        else:
            st.info("No significant keywords found in notes.")


# --- Displaying the raw data table at the bottom ---
st.markdown("---")
st.subheader("Truck Fleet Overview")
st.dataframe(df, use_container_width=True)
