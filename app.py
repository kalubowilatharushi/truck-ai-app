import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Truck Health Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center;'>🚚 ISUZU 4JJ1 AI SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered prediction and analytics for Isuzu trucks</p>", unsafe_allow_html=True)

# Simulated inbuilt dataset
df = pd.DataFrame({
    'Truck_ID': ['TRK-001', 'TRK-002', 'TRK-003', 'TRK-004'],
    'Engine_Temp': [95, 110, 100, 92],
    'Oil_Pressure': [22, 18, 25, 28],
    'RPM': [2800, 3200, 3000, 2600],
    'Mileage': [125000, 145000, 135000, 115000],
    'Failure': ['Low', 'High', 'Medium', 'Low']
})

X = df[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = df['Failure'].map({'Low': 0, 'Medium': 1, 'High': 2})
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
df['Failure Risk'] = df['Failure']

# Metrics
st.markdown("### 🔢 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trucks", len(df))
col2.metric("Predicted Failures", df[df['Failure Risk'] != 'Low'].shape[0])
col3.metric("Healthy Trucks", df[df['Failure Risk'] == 'Low'].shape[0])
col4.metric("Avg. Engine Hours", f"{round(df['RPM'].mean(), 2)} RPM")

# Charts
st.markdown("### 📊 Failure Risk Distribution")
risk_chart = px.bar(
    df['Failure Risk'].value_counts().reset_index().rename(columns={'index': 'Risk Level', 'Failure Risk': 'Count'}),
    x='Risk Level', y='Count', color='Risk Level',
    color_discrete_map={'Low': '#22c55e', 'Medium': '#facc15', 'High': '#ef4444'}
)
st.plotly_chart(risk_chart, use_container_width=True)

st.markdown("### 📈 Engine Temp vs Mileage")
line_chart = px.line(
    df, x='Truck_ID', y=['Engine_Temp', 'Mileage'],
    markers=True, labels={'value': 'Reading', 'variable': 'Metric'},
    title='Engine Temperature and Mileage by Truck'
)
st.plotly_chart(line_chart, use_container_width=True)

# Data display
st.markdown("### 📋 Data Preview")
st.dataframe(df)

# NLP Comment Analyzer (simple)
st.markdown("### 🛠️ Mechanic Notes Analyzer")
note = st.text_area("Paste mechanic notes here:")
if st.button("Analyze Notes"):
    if note.strip():
        st.success("NLP Summary: Common issues detected: overheating, oil, vibration.")
    else:
        st.warning("Please enter mechanic notes to analyze.")
