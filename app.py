import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import io
from PIL import Image

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Truck AI Health", layout="wide")

# ----------------- CSS for clean UI -----------------
st.markdown("""
    <style>
        body { font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #1e1e2f; color: white; }
        .css-18e3th9 { background-color: #1e1e2f; }
        .css-1d391kg { color: white; }
    </style>
""", unsafe_allow_html=True)

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("inbuilt_truck_data.csv")
    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = "Low"
    if "Possible_Issue" not in df.columns:
        df["Possible_Issue"] = "Normal"
    return df

df = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🧭 Navigate")
tab = st.sidebar.radio("Go to", ["Dashboard", "Note Analyzer", "Report"])

# ----------------- TAB 1: DASHBOARD -----------------
if tab == "Dashboard":
    st.markdown("## 🚛 Truck AI - Health Prediction Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trucks", len(df))
    col2.metric("High Risk Trucks", sum(df["Risk_Level"] == "High"))
    col3.metric("Failures Detected", sum(df["Possible_Issue"] != "Normal"))

    # Pie chart of issue types
    st.subheader("❌ Failure Cause Distribution")
    issue_counts = df["Possible_Issue"].value_counts().head(6)
    fig1, ax1 = plt.subplots()
    ax1.pie(issue_counts.values, labels=issue_counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)

    # Bar chart
    st.subheader("🔧 Failure Cause Frequency")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, y="Possible_Issue", order=df["Possible_Issue"].value_counts().index, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📋 Truck Data with Issue Analysis")
    st.dataframe(df.head(100), use_container_width=True)

# ----------------- TAB 2: NOTE ANALYZER -----------------
elif tab == "Note Analyzer":
    st.markdown("## 🛠️ Mechanic Note Analyzer")
    comment = st.text_area("Paste mechanic notes:")
    if st.button("Analyze Notes"):
        if not comment.strip():
            st.warning("⚠️ Please enter a mechanic comment before analyzing.")
        else:
            st.success("✅ Comment received.")
            sample_comments = [
                "Engine overheating, coolant leaking",
                "Oil pressure dropping, possible leak",
                "Knocking sound from engine",
                "Oil filter needs replacement",
                "Vibration at high RPM",
                "Coolant low again",
                "RPM unstable at idle"
            ]
            sample_comments.append(comment)

            vectorizer = TfidfVectorizer(stop_words='english')
            X_text = vectorizer.fit_transform(sample_comments)
            features = vectorizer.get_feature_names_out()
            scores = X_text.toarray().sum(axis=0)
            keywords = sorted(dict(zip(features, scores)).items(), key=lambda x: x[1], reverse=True)[:5]

            st.subheader("🔍 Top Keywords")
            for word, score in keywords:
                st.write(f"- {word} (Score: {score:.2f})")

            st.subheader("🧠 AI-Based Recommendations")
            suggestions = []
            steps = []

            if "oil" in comment.lower():
                suggestions.append("Check the oil system for leaks or pressure drops.")
                steps.append("1. Locate oil filter.\n2. Inspect for leaks.\n3. Replace if clogged.")
            if "coolant" in comment.lower():
                suggestions.append("Inspect the cooling system and radiator cap.")
                steps.append("1. Refill coolant.\n2. Inspect hoses.\n3. Replace radiator cap.")
            if "vibration" in comment.lower():
                suggestions.append("Investigate possible engine vibration causes.")
                steps.append("1. Inspect mounts.\n2. Check spark plugs.\n3. Balance components.")

            for s in suggestions:
                st.info(s)
            if steps:
                st.subheader("🧾 Guided Fix Steps")
                for s in steps:
                    st.text(s)

            st.session_state.analysis_keywords = keywords
            st.session_state.analysis_recs = suggestions
            st.session_state.analysis_steps = steps

# ----------------- TAB 3: REPORT -----------------
elif tab == "Report":
    st.markdown("## 📄 Download Truck Maintenance Report")

    def generate_pdf(keywords_list, ai_recs, steps):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Truck AI - Maintenance Report", ln=True, align="C")

        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Top Keywords:", ln=True)
        for word, score in keywords_list:
            pdf.cell(200, 10, txt=f"- {word}: {score:.2f}", ln=True)

        pdf.ln(5)
        pdf.cell(200, 10, txt="AI Recommendations:", ln=True)
        for rec in ai_recs:
            pdf.multi_cell(200, 10, txt=f"- {rec}")

        pdf.ln(5)
        pdf.cell(200, 10, txt="Suggested Fix Steps:", ln=True)
        for s in steps:
            pdf.multi_cell(200, 10, txt=s)

        return pdf.output(dest='S').encode('latin-1')

    if "analysis_keywords" in st.session_state:
        if st.button("📥 Download PDF Report"):
            pdf_bytes = generate_pdf(
                st.session_state.analysis_keywords,
                st.session_state.analysis_recs,
                st.session_state.analysis_steps
            )
            st.download_button(
                label="📄 Download Report",
                data=pdf_bytes,
                file_name="truck_ai_report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("⚠️ Run a note analysis first to generate report.")
