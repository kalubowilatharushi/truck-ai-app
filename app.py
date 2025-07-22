import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import os

# -------------- UI SETTINGS --------------
st.set_page_config(page_title="Truck AI - Health Prediction", layout="wide")
st.markdown("<style>body{color:#fff;background-color:#1e1e2f;}</style>", unsafe_allow_html=True)

# -------------- LOAD DATA --------------
@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")  # CSV placed in root with app.py

df = load_data()

# -------------- BASIC DATA PREP --------------
df["Risk_Level"] = df["Risk_Level"].fillna("Low")
df["Possible_Issue"] = df["Possible_Issue"].fillna("Normal")

# Simplify complex causes to clean set
def simplify_issue(issue):
    simplified = {
        "Aging Components": "Aging",
        "Oil System": "Oil",
        "Cooling System": "Cooling",
        "Spark Plug": "Spark",
        "Engine Vibrations": "Engine"
    }
    for key, value in simplified.items():
        if key in issue:
            return value
    return "Normal"

df["Simplified_Issue"] = df["Possible_Issue"].apply(simplify_issue)

# -------------- SIDEBAR NAVIGATION --------------
menu = st.sidebar.radio("Navigate", ["📊 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# -------------- DASHBOARD TAB --------------
if menu == "📊 Dashboard":
    st.title("🚛 Truck AI - Health Prediction Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trucks", len(df))
    col2.metric("High Risk Trucks", sum(df["Risk_Level"] == "High"))
    col3.metric("Failures Detected", sum(df["Simplified_Issue"] != "Normal"))

    st.subheader("❌ Failure Cause Distribution")
    pie_data = df["Simplified_Issue"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data.values, labels=pie_data.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    st.subheader("🔧 Failure Cause Frequency")
    fig2, ax2 = plt.subplots()
    df["Simplified_Issue"].value_counts().plot(kind="bar", color="coral", ax=ax2)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Possible Issue")
    st.pyplot(fig2)

    st.subheader("📋 Truck Data with Issue Analysis")
    st.dataframe(df[["Engine_Temp", "Oil_Pressure", "RPM", "Mileage", "Risk_Level", "Possible_Issue"]])

# -------------- NOTE ANALYZER TAB --------------
elif menu == "🛠️ Note Analyzer":
    st.title("🛠️ Mechanic Note Analyzer")

    input_note = st.text_area("Paste mechanic notes:")
    if st.button("Analyze"):
        if not input_note.strip():
            st.warning("⚠️ Please enter a comment to analyze.")
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([input_note])
            keywords = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray().flatten()

            sorted_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:5]
            st.markdown("### 🔍 Top Keywords")
            for word, score in sorted_keywords:
                st.markdown(f"- `{word}` (Score: {score:.2f})")

            st.markdown("### 🤖 AI-Based Recommendations")
            ai_rec = []
            for word in keywords:
                if "filter" in word:
                    ai_rec.append("Check oil filter for clogs.")
                elif "noise" in word:
                    ai_rec.append("Inspect engine mounts for vibrations.")
                elif "smoke" in word:
                    ai_rec.append("Check for oil/coolant leakage.")
            if not ai_rec:
                ai_rec.append("General inspection advised.")
            for rec in ai_rec:
                st.markdown(f"✅ {rec}")

            # Save session state
            st.session_state.keywords = sorted_keywords
            st.session_state.recs = ai_rec

# -------------- REPORT TAB --------------
elif menu == "📄 Report":
    st.title("📄 Generate Maintenance Report")

    if "keywords" in st.session_state and "recs" in st.session_state:
        st.success("✅ Analysis data found. Ready to export.")

        # PDF generator
        def generate_pdf(keywords, recs):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Truck AI - Maintenance Report", ln=True, align="C")

            pdf.ln(10)
            pdf.cell(200, 10, txt="Top Keywords:", ln=True)
            for k, s in keywords:
                pdf.cell(200, 8, txt=f"- {k}: {s:.2f}", ln=True)

            pdf.ln(10)
            pdf.cell(200, 10, txt="AI Recommendations:", ln=True)
            for rec in recs:
                pdf.multi_cell(0, 8, txt=f"- {rec}")

            # Save
            pdf.output("maintenance_report.pdf")

        if st.button("📥 Download PDF Report"):
            generate_pdf(st.session_state.keywords, st.session_state.recs)
            with open("maintenance_report.pdf", "rb") as f:
                st.download_button(
                    label="⬇️ Click here to download",
                    data=f,
                    file_name="truck_maintenance_report.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("⚠️ No mechanic note analysis found. Please go to the Note Analyzer tab first.")
