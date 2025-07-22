import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import random

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# ---------- Inbuilt Data (1000 records)
@st.cache_data
def load_data():
    import numpy as np
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        'Engine_Temp': rng.integers(70, 110, 1000),
        'Oil_Pressure': rng.integers(15, 35, 1000),
        'RPM': rng.integers(1800, 3500, 1000),
        'Mileage': rng.integers(90000, 180000, 1000)
    })
    # Risk Level
    def classify(row):
        score = row['Engine_Temp'] + (40 - row['Oil_Pressure']) + (row['RPM']//100) + (row['Mileage']//10000)
        if score < 200: return 0
        elif score < 250: return 1
        else: return 2
    df['Failure'] = df.apply(classify, axis=1)
    return df

data = load_data()

# ---------- Train Model
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)
labels = {0: 'Low', 1: 'Medium', 2: 'High'}

# ---------- Sidebar Navigation
tabs = st.sidebar.radio("Navigation", ["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# ---------- Dashboard
if tabs == "🏠 Dashboard":
    st.title("🚚 AI - Truck Health Dashboard")

    # Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Fleet Health", "92%", "+2% from last month")
    col2.metric("Active Alerts", "4", "1 critical")
    col3.metric("Uptime Increase", "15%", "+3%")
    col4.metric("Downtime Reduction", "25%", "+5%")

    # Prediction Trends Chart
    st.subheader("📈 Failure Prediction Trends")
    fig, ax = plt.subplots()
    weekly_counts = data.sample(300).copy()
    weekly_counts['week'] = [f"Week {i%12+1}" for i in range(300)]
    trend_data = weekly_counts.groupby(['week', 'Failure']).size().unstack(fill_value=0).sort_index()
    trend_data.rename(columns=labels, inplace=True)
    trend_data.plot(kind='bar', stacked=True, ax=ax, colormap="cool")
    st.pyplot(fig)

    # Pie Chart
    st.subheader("📊 Failure Types")
    failure_counts = data['Failure'].map(labels).value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

    # Vehicle Status
    st.subheader("🚛 Vehicle Status")
    for i, label in enumerate(["TRK-001", "TRK-007", "TRK-004", "TRK-002"]):
        status = ["Operational", "Maintenance", "Operational", "At Risk"]
        percent = [80, 25, 60, 45]
        st.progress(percent[i])
        st.write(f"**{label}** – {status[i]} ({percent[i]}%)")

    # Recent Activity
    st.subheader("🕒 Recent Activity")
    st.markdown("""
    - ✅ Generated report for Q2 performance (5 mins ago)  
    - 🔍 Analyzed maintenance log for TRK-007 (2 hrs ago)  
    - ⚠️ Critical alert triggered for TRK-001 (8 hrs ago)  
    - 🛠️ Cleared warning alert for TRK-004 (1 day ago)  
    """)

# ---------- Note Analyzer
elif tabs == "🛠️ Note Analyzer":
    st.title("🛠️ Mechanic Notes Analyzer with AI")

    user_note = st.text_area("Enter mechanic note:")
    if st.button("Analyze Comment"):
        if user_note.strip() == "":
            st.warning("⚠️ Please enter a comment before submitting.")
        else:
            # TF-IDF NLP
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform([user_note])
            keywords = vectorizer.get_feature_names_out()
            scores = tfidf.toarray().flatten()
            sorted_kw = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:5]

            st.success("🔍 Analysis Complete!")
            st.write("**Top Keywords:**")
            for word, score in sorted_kw:
                st.write(f"- `{word}` (Score: {score:.2f})")

            # AI Suggestions (Sample rules)
            ai_recs, steps = [], []
            if "oil" in user_note or "pressure" in user_note:
                ai_recs.append("Check oil pump and filter")
                steps.append("1. Locate oil pump\n2. Inspect for clogs\n3. Replace filter if needed")
            if "engine" in user_note or "temperature" in user_note:
                ai_recs.append("Inspect cooling system")
                steps.append("1. Check coolant level\n2. Inspect radiator and fan")

            st.write("### 🤖 AI Recommendations:")
            for rec in ai_recs:
                st.markdown(f"- {rec}")

            st.write("### 📋 Guided Troubleshooting Steps:")
            for s in steps:
                st.markdown(f"- {s}")

            # Save to session
            st.session_state['report_keywords'] = sorted_kw
            st.session_state['report_recs'] = ai_recs
            st.session_state['report_steps'] = steps

# ---------- Report Tab
elif tabs == "📄 Report":
    st.title("📄 Maintenance Report Generator")

    if "report_keywords" not in st.session_state:
        st.info("🔍 No data to generate report. Please analyze a note first.")
    else:
        if st.button("📥 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Truck AI Maintenance Report", ln=True, align="C")
            pdf.cell(200, 10, txt=str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")), ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Extracted Keywords:", ln=True)
            pdf.set_font("Arial", size=12)
            for word, score in st.session_state['report_keywords']:
                pdf.cell(200, 10, txt=f"- {word} ({score:.2f})", ln=True)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="AI Recommendations:", ln=True)
            pdf.set_font("Arial", size=12)
            for rec in st.session_state['report_recs']:
                pdf.multi_cell(0, 10, f"- {rec}")

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Guided Steps:", ln=True)
            pdf.set_font("Arial", size=12)
            for step in st.session_state['report_steps']:
                pdf.multi_cell(0, 10, f"- {step}")

            file_path = "truck_report.pdf"
            pdf.output(file_path)
            with open(file_path, "rb") as f:
                st.download_button("📤 Download Report", f, file_name="truck_report.pdf", mime="application/pdf")
