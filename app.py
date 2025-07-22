import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime
import numpy as np

# Page configuration
st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# --------- Load Inbuilt Data
@st.cache_data
def load_data():
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        'Engine_Temp': rng.integers(70, 110, 1000),
        'Oil_Pressure': rng.integers(15, 35, 1000),
        'RPM': rng.integers(1800, 3500, 1000),
        'Mileage': rng.integers(90000, 180000, 1000)
    })
    def classify(row):
        score = row['Engine_Temp'] + (40 - row['Oil_Pressure']) + (row['RPM']//100) + (row['Mileage']//10000)
        if score < 200: return 0
        elif score < 250: return 1
        else: return 2
    df['Failure'] = df.apply(classify, axis=1)
    return df

data = load_data()
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)
labels = {0: 'Low', 1: 'Medium', 2: 'High'}

# --------- Sidebar Navigation
tabs = st.sidebar.radio("📁 Navigation", ["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# --------- Dashboard Tab
if tabs == "🏠 Dashboard":
    st.title("🚚 AI - Truck Health Dashboard")

    # Metrics Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fleet Health", "92%", "+2%")
    col2.metric("Active Alerts", "4", "1 critical")
    col3.metric("Uptime", "15%", "+3%")
    col4.metric("Downtime ↓", "25%", "-5%")

    # Failure Trend Bar Chart
    st.subheader("📈 Failure Prediction Trends")
    weekly_sample = data.sample(300).copy()
    weekly_sample['week'] = [f"Week {i%12+1}" for i in range(300)]
    trend_data = weekly_sample.groupby(['week', 'Failure']).size().unstack(fill_value=0).sort_index()
    trend_data.rename(columns=labels, inplace=True)

    fig1, ax1 = plt.subplots()
    trend_data.plot(kind='bar', stacked=True, ax=ax1, colormap="coolwarm")
    st.pyplot(fig1)

    # Failure Type Pie Chart
    st.subheader("📊 Failure Types")
    failure_counts = data['Failure'].map(labels).value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(failure_counts, labels=failure_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

    # Vehicle Progress
    st.subheader("🚛 Vehicle Statuses")
    for i, truck in enumerate(["TRK-001", "TRK-007", "TRK-004", "TRK-002"]):
        progress = [80, 25, 60, 45][i]
        st.progress(progress)
        st.write(f"**{truck}** – {'At Risk' if progress < 50 else 'Operational'} ({progress}%)")

    # Activity Feed
    st.subheader("🕒 Recent Activity")
    st.markdown("""
    - ✅ Report generated (5 mins ago)  
    - 🔍 Log analyzed: TRK-007 (2 hrs ago)  
    - ⚠️ Alert for TRK-001 (8 hrs ago)  
    - 🛠️ Warning cleared: TRK-004 (1 day ago)  
    """)

# --------- NLP Analyzer Tab
elif tabs == "🛠️ Note Analyzer":
    st.title("🛠️ Mechanic Notes Analyzer")

    user_input = st.text_area("Enter mechanic notes here:")
    if st.button("Analyze Note"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a comment before submitting.")
        else:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform([user_input])
            features = vectorizer.get_feature_names_out()
            scores = tfidf.toarray().flatten()
            top_keywords = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)[:5]

            st.success("✅ Analysis complete!")
            st.subheader("🔑 Top Keywords")
            for word, score in top_keywords:
                st.markdown(f"- `{word}` (Score: {score:.2f})")

            # AI Suggestions (simple logic-based)
            ai_recs, steps = [], []
            if "oil" in user_input or "pressure" in user_input:
                ai_recs.append("🔧 Check oil pressure system")
                steps.append("1. Inspect oil pump\n2. Replace filter\n3. Refill oil")
            if "engine" in user_input or "heat" in user_input:
                ai_recs.append("🌡️ Check engine cooling")
                steps.append("1. Check coolant level\n2. Inspect radiator\n3. Replace fan if faulty")

            st.subheader("🤖 AI Recommendations")
            for rec in ai_recs:
                st.markdown(f"- {rec}")

            st.subheader("🧭 Troubleshooting Steps")
            for step in steps:
                st.markdown(f"- {step}")

            # Save to session state
            st.session_state['keywords'] = top_keywords
            st.session_state['recs'] = ai_recs
            st.session_state['steps'] = steps

# --------- Report Tab
elif tabs == "📄 Report":
    st.title("📄 Generate PDF Report")

    if "keywords" not in st.session_state:
        st.info("ℹ️ No analysis yet. Please use the Note Analyzer first.")
    else:
        if st.button("📤 Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, "Truck AI Maintenance Report", ln=True, align="C")
            pdf.cell(200, 10, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Top Keywords:", ln=True)
            pdf.set_font("Arial", size=12)
            for word, score in st.session_state['keywords']:
                pdf.cell(200, 10, f"- {word} ({score:.2f})", ln=True)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "AI Recommendations:", ln=True)
            pdf.set_font("Arial", size=12)
            for r in st.session_state['recs']:
                pdf.multi_cell(0, 10, f"- {r}")

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Troubleshooting Steps:", ln=True)
            pdf.set_font("Arial", size=12)
            for s in st.session_state['steps']:
                pdf.multi_cell(0, 10, f"- {s}")

            pdf.output("truck_report.pdf")
            with open("truck_report.pdf", "rb") as f:
                st.download_button("📥 Download Report", f, file_name="truck_report.pdf", mime="application/pdf")
