import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (run once)
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Truck Health Prediction",
    page_icon="🚚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1r6slb0 {  /* Metric container */
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🚚 Navigation")
    page = st.radio("", ["Dashboard", "Predictions", "Analysis"])

# Main content
st.title("Truck Health Prediction System")

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trucks", "150", "+5")
with col2:
    st.metric("High Risk", "12", "-2")
with col3:
    st.metric("Medium Risk", "28", "+3")
with col4:
    st.metric("Low Risk", "110", "+4")

if page == "Dashboard":
    # Risk Distribution Chart
    risk_data = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Count': [110, 28, 12]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Distribution")
        fig = px.pie(risk_data, values='Count', names='Risk Level',
                    color_discrete_sequence=['green', 'yellow', 'red'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Recent Alerts")
        st.warning("⚠️ Truck ID 123: High oil pressure")
        st.info("ℹ️ Truck ID 456: Scheduled maintenance due")

elif page == "Predictions":
    st.markdown("### Engine Health Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        engine_hours = st.number_input("Engine Hours", 0, 10000)
        oil_pressure = st.number_input("Oil Pressure (PSI)", 0.0, 100.0)
    with col2:
        temperature = st.number_input("Temperature (°C)", 0.0, 150.0)
        vibration = st.number_input("Vibration Level", 0.0, 10.0)

    if st.button("Predict"):
        # Simple demo prediction logic
        if temperature > 90 or vibration > 7:
            st.error("🔴 High Risk: Immediate attention required")
        elif temperature > 70 or vibration > 5:
            st.warning("🟡 Medium Risk: Monitor closely")
        else:
            st.success("🟢 Low Risk: Normal operation")

elif page == "Analysis":
    st.markdown("### Mechanic Notes Analysis")
    
    notes = st.text_area("Enter mechanic notes for analysis")
    
    if st.button("Analyze"):
        if notes:
            # Basic NLP analysis
            tokens = word_tokenize(notes.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            word_freq = Counter(tokens)
            
            # Display results
            st.markdown("#### Top Keywords Found:")
            for word, count in word_freq.most_common(5):
                st.markdown(f"- {word}: {count} times")
            
            # Simple sentiment analysis
            maintenance_terms = ['repair', 'replace', 'broken', 'issue', 'problem']
            if any(term in notes.lower() for term in maintenance_terms):
                st.warning("⚠️ Maintenance may be required")
            else:
                st.success("✅ No immediate issues detected")

# File upload section at the bottom
st.markdown("---")
st.markdown("### Data Upload")
uploaded_file = st.file_uploader("Upload maintenance records", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
