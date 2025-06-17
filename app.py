import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="ISUZU 4JJ1 AI SYSTEM", layout="wide")

# Load inbuilt data
@st.cache_data
def load_data():
    df = pd.read_csv("inbuilt_truck_data.csv")
    return df

df = load_data()

# Train model
X = df[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = df['Failure']
model = RandomForestClassifier().fit(X, y)
labels = {0: '🟢 Low', 1: '🟡 Medium', 2: '🔴 High'}

# Predict
df["Risk"] = [labels[p] for p in model.predict(X)]

# Sidebar
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Truck Health Prediction", "🛠️ Notes Analyzer", "📁 Reports"])

if page == "📊 Dashboard":
    st.title("🚚 ISUZU 4JJ1 AI SYSTEM")
    st.markdown("### Predictive Maintenance Overview")

    total = len(df)
    high = sum(df["Risk"] == "🔴 High")
    medium = sum(df["Risk"] == "🟡 Medium")
    low = sum(df["Risk"] == "🟢 Low")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trucks", total)
    col2.metric("🔴 High Risk", high)
    col3.metric("🟡 Medium Risk", medium)
    col4.metric("🟢 Low Risk", low)

elif page == "🔍 Truck Health Prediction":
    st.title("🔍 Truck Health Prediction")

    st.dataframe(df[["Engine_Temp", "Oil_Pressure", "RPM", "Mileage", "Risk"]])

elif page == "🛠️ Notes Analyzer":
    st.title("🛠️ Maintenance Notes Analyzer")

    text_input = st.text_area("Paste technician notes here:")
    if st.button("Analyze Notes"):
        if text_input.strip() == "":
            st.warning("⚠️ Please enter some notes.")
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            X_text = vectorizer.fit_transform([text_input])
            features = vectorizer.get_feature_names_out()
            scores = X_text.toarray().flatten()
            keywords = sorted(zip(features, scores), key=lambda x: -x[1])[:5]

            st.markdown("### 🔍 Top Keywords")
            for word, score in keywords:
                st.markdown(f"- `{word}` (Score: {score:.2f})")

            st.markdown("### 🧰 Suggested Action")
            st.info("Inspect cooling system, throttle response, and oil quality.")

elif page == "📁 Reports":
    st.title("📁 Maintenance Reports")

    report_data = [
        ["2025-06-10", "TRK-001", "Overheating engine", "Critical", "Replaced coolant system"],
        ["2025-06-12", "TRK-002", "Oil pressure drop", "Warning", "Topped up oil and cleaned pump"],
    ]
    report_df = pd.DataFrame(report_data, columns=["Date", "Truck ID", "Problem Summary", "Status", "Action Taken"])
    st.dataframe(report_df)
    st.download_button("📥 Download Report", report_df.to_csv(index=False), "maintenance_report.csv", "text/csv")
