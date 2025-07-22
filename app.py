import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# ---------- Load Inbuilt Dataset (1000 samples)
@st.cache_data
def load_data():
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        'Engine_Temp': rng.integers(70, 110, 1000),
        'Oil_Pressure': rng.integers(15, 35, 1000),
        'RPM': rng.integers(1800, 3500, 1000),
        'Mileage': rng.integers(90000, 180000, 1000)
    })
    return df

data = load_data()

# ---------- Classify Failure Risk
def classify(row):
    score = row['Engine_Temp'] + (40 - row['Oil_Pressure']) + (row['RPM'] // 100) + (row['Mileage'] // 10000)
    if score < 200:
        return 0
    elif score < 250:
        return 1
    else:
        return 2

data['Failure'] = data.apply(classify, axis=1)
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
data['Risk_Level'] = data['Failure'].map(labels)

# ---------- Detect Possible Part Failures
def detect_issue(row):
    issues = []
    if row['Engine_Temp'] > 100:
        issues.append("Cooling System")
    if row['Oil_Pressure'] < 20:
        issues.append("Oil System")
    if row['RPM'] > 3000 and row['Engine_Temp'] > 95:
        issues.append("Spark Plug / Engine Vibrations")
    if row['Mileage'] > 140000:
        issues.append("Aging Components")
    return ", ".join(issues) if issues else "Normal"

data['Possible_Issue'] = data.apply(detect_issue, axis=1)

# ---------- Train Model
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

# ---------- Navigation
page = st.sidebar.radio("Navigate", ["Dashboard", "Note Analyzer", "Report"])

# ---------- Dashboard
if page == "Dashboard":
    st.title("🚛 Truck AI - Health Prediction Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trucks", len(data))
    col2.metric("High Risk Trucks", (data['Failure'] == 2).sum())
    col3.metric("Failures Detected", (data['Possible_Issue'] != "Normal").sum())

    # Pie Chart
    st.subheader("🧩 Failure Cause Distribution")
    pie_data = data['Possible_Issue'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    ax1.axis('equal')
    st.pyplot(fig1)

    # Bar Chart
    st.subheader("🔧 Failure Cause Frequency")
    fig2, ax2 = plt.subplots()
    pie_data.plot(kind='bar', color='tomato', ax=ax2)
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # Data Table
    st.subheader("📋 Truck Data with Issue Analysis")
    st.dataframe(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage', 'Risk_Level', 'Possible_Issue']])

# ---------- Note Analyzer
elif page == "Note Analyzer":
    st.title("🛠️ Mechanic Note Analyzer")
    text = st.text_area("Paste mechanic notes:")
    if st.button("Analyze"):
        if not text.strip():
            st.warning("⚠️ Please enter mechanic notes to analyze.")
        else:
            tfidf = TfidfVectorizer(stop_words='english')
            matrix = tfidf.fit_transform([text])
            keywords = tfidf.get_feature_names_out()
            scores = matrix.toarray()[0]
            top_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:5]
            st.write("**Top Keywords:**")
            for word, score in top_keywords:
                st.write(f"- {word} ({score:.2f})")

            # AI Suggestions
            recs, steps = [], []
            for word, _ in top_keywords:
                if "oil" in word:
                    recs.append("🔧 Oil pressure issue suspected.")
                    steps.append("1. Check oil level\n2. Inspect oil filter\n3. Replace oil if necessary")
                elif "engine" in word:
                    recs.append("🚨 Engine overheating or knocking signs.")
                    steps.append("1. Inspect engine block\n2. Test coolant system\n3. Check head gasket")
                elif "coolant" in word:
                    recs.append("💧 Coolant problem detected.")
                    steps.append("1. Top-up coolant\n2. Inspect radiator\n3. Replace coolant cap")
                elif "vibration" in word or "knocking" in word:
                    recs.append("⚙️ Engine misfire or spark plug issue.")
                    steps.append("1. Replace spark plugs\n2. Inspect engine mountings")

            if recs:
                st.markdown("---")
                st.write("### 🤖 AI Suggestions:")
                for r in recs:
                    st.markdown(f"- {r}")
                st.markdown("### 🧰 Maintenance Steps:")
                for s in steps:
                    st.markdown(f"- {s}")

            st.session_state['keywords'] = top_keywords
            st.session_state['notes'] = text
            st.session_state['recs'] = recs
            st.session_state['steps'] = steps

# ---------- Report Generator
elif page == "Report":
    st.title("📄 Generate Maintenance Report")
    if 'keywords' not in st.session_state:
        st.info("Please run the Note Analyzer first.")
    else:
        if st.button("📥 Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Truck Maintenance Report", ln=True, align="C")
            pdf.ln(10)
            pdf.multi_cell(0, 10, "Mechanic Notes:")
            pdf.multi_cell(0, 10, st.session_state['notes'])

            pdf.ln(5)
            pdf.cell(200, 10, "Top Keywords:", ln=True)
            for word, score in st.session_state['keywords']:
                pdf.cell(200, 10, f"- {word} ({score:.2f})", ln=True)

            pdf.ln(5)
            pdf.cell(200, 10, "AI Recommendations:", ln=True)
            for r in st.session_state['recs']:
                pdf.multi_cell(0, 10, r)

            pdf.ln(5)
            pdf.cell(200, 10, "Maintenance Steps:", ln=True)
            for s in st.session_state['steps']:
                pdf.multi_cell(0, 10, s)

            filename = "truck_maintenance_report.pdf"
            pdf.output(filename)
            with open(filename, "rb") as f:
                st.download_button("📥 Download Report", f, file_name=filename, mime="application/pdf")
