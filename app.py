Here's the correct code without any comments at the beginning. Save this exactly as shown below as `app.py`:

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Truck Health Prediction",
    page_icon="🚚",
    layout="wide"
)

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
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🚚 Navigation")
    page = st.radio("", ["Dashboard", "Predictions", "Analysis"])

st.title("Truck Health Prediction System")

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
            words = notes.lower().split()
            word_count = len(words)
            
            maintenance_terms = ['repair', 'replace', 'broken', 'issue', 'problem']
            found_terms = [word for word in words if word in maintenance_terms]
            
            st.markdown(f"Word count: {word_count}")
            if found_terms:
                st.warning(f"Found maintenance keywords: {', '.join(found_terms)}")
            else:
                st.success("No immediate issues detected")

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
```

Make sure to:
1. Install required packages:
```bash
pip install streamlit pandas plotly
```

2. Run the app:
```bash
streamlit run app.py
```

This should now work without any syntax errors. Would you like me to explain any part of the code?
