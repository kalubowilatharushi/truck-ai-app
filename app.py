I'll provide an even simpler version that doesn't require NLTK. Here's a streamlined version that will work with minimal dependencies:

```python
import streamlit as st
import pandas as pd
import plotly.express as px

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
        # Simple prediction logic
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
            # Simple analysis
            words = notes.lower().split()
            word_count = len(words)
            
            # Basic keyword checking
            maintenance_terms = ['repair', 'replace', 'broken', 'issue', 'problem']
            found_terms = [word for word in words if word in maintenance_terms]
            
            st.markdown(f"Word count: {word_count}")
            if found_terms:
                st.warning(f"Found maintenance keywords: {', '.join(found_terms)}")
            else:
                st.success("No immediate issues detected")

# File upload section
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

To use this version:

1. Install only the essential packages:
```bash
pip install streamlit pandas plotly
```

2. Save this code as `app.py`

3. Run the app:
```bash
streamlit run app.py
```

This version:
- Removes NLTK dependency
- Keeps core functionality
- Maintains professional appearance
- Uses simpler text analysis
- Works with minimal setup

Would you like me to explain any part or make additional simplifications?
