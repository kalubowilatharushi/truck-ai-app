import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import base64

# --- Setup ---
st.set_page_config(page_title="Truck AI System", layout="wide")
st.markdown("""
    <style>
        .main {background-color: #1c1c1e;}
        .block-container {padding: 2rem;}
        .stTabs [role="tablist"] {background: #333;}
        .stTabs [role="tab"] {color: white;}
        h1, h2, h3, h4 {color: #fff;}
    </style>
""", unsafe_allow_html=True)

# --- Inbuilt 1000 Record Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/kalubowilatharushi/truck-ai-app/main/data/inbuilt_truck_data.csv")
    return df

# --- ML Model Training ---
@st.cache_resource
def train_model(data):
    model = RandomForestClassifier()
    model.fit(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']], data['Risk_Level'])
    return model

df = load_data()
model = train_model(df)

# --- Failure Issue Mapping ---
issue_map = {
    "Low": "Normal",
    "Medium": "Cooling System",
    "High": "Oil System, Aging Components, Engine Vibrations, Spark Plug"
}
df["Possible_Issue"] = df["Risk_Level"].map(issue_map)

# --- Sidebar Navigation ---
tab = st.sidebar.radio("Navigate", ["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# ========== 🏠 Dashboard Tab ==========
if tab == "🏠 Dashboard":
    st.markdown("## 🚚 Truck AI - Health Prediction Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trucks", len(df))
    col2.metric("High Risk Trucks", (df["Risk_Level"] == "High").sum())
    col3.metric("Failures Detected", (df["Risk_Level"] != "Low").sum())

    # Failure Cause Distribution (Top 5 + Others)
    issue_counts = df["Possible_Issue"].str.split(", ").explode().value_counts()
    top_issues = issue_counts[:5]
    others = issue_counts[5:].sum()
    pie_data = top_issues.append(pd.Series({"Others": others}))

    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    ax1.set_title("Failure Cause Distribution")
    st.pyplot(fig1)

    # Failure Frequency Chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    top_failures = df["Possible_Issue"].value_counts().head(7)
    ax2.bar(top_failures.index, top_failures.values, color='tomato')
    ax2.set_xticklabels(top_failures.index, rotation=30)
    ax2.set_ylabel("Count")
    ax2.set_title("Failure Cause Frequency")
    st.pyplot(fig2)

    # Raw Table
    st.markdown("### 🧾 Truck Data with Issue Analysis")
    st.dataframe(df.head(50), use_container_width=True)

# ========== 🛠️ Note Analyzer ==========
elif tab == "🛠️ Note Analyzer":
    st.markdown("## 🛠️ Mechanic Note Analyzer")
    note = st.text_area("Paste mechanic notes:")
    if st.button("Analyze Notes"):
        if not note.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            logs = [note]
            vectorizer = TfidfVectorizer(stop_words='english')
            X_text = vectorizer.fit_transform(logs)
            feature_names = vectorizer.get_feature_names_out()
            scores = X_text.toarray().flatten()
            keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:5]

            st.markdown("### 🔍 Top Keywords")
            for word, score in keywords:
                st.markdown(f"- `{word}` (score: {score:.2f})")

            # AI Recommendations (simple example)
            recommendations = []
            steps = []
            if "oil" in note.lower():
                recommendations.append("🔧 Inspect oil filter and oil levels.")
                steps.append("1. Locate oil filter.\n2. Inspect for leaks.\n3. Replace if clogged.")
            if "engine" in note.lower():
                recommendations.append("⚙️ Run full engine diagnostics.")
                steps.append("1. Connect diagnostic tool.\n2. Run scan.\n3. Review logs.")
            if "coolant" in note.lower():
                recommendations.append("🧊 Check coolant levels and radiator cap.")
                steps.append("1. Open coolant tank.\n2. Check for low fluid.\n3. Inspect cap pressure.")

            st.markdown("### 🤖 AI Recommendations")
            for rec in recommendations:
                st.info(rec)

            st.markdown("### 📝 Guided Steps")
            for step in steps:
                st.code(step, language="markdown")

            # Store for PDF
            st.session_state["report_note"] = note
            st.session_state["report_keywords"] = keywords
            st.session_state["report_ai"] = recommendations
            st.session_state["report_steps"] = steps

# ========== 📄 Report ==========
elif tab == "📄 Report":
    st.markdown("## 📄 Download Full Maintenance Report")

    if "report_note" in st.session_state:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="🚚 AI Maintenance Report", ln=True, align="C")
        pdf.cell(200, 10, txt="", ln=True)

        pdf.cell(200, 10, txt="📝 Mechanic Notes:", ln=True)
        pdf.multi_cell(0, 10, txt=st.session_state["report_note"])

        pdf.cell(200, 10, txt="🔍 Top Keywords:", ln=True)
        for word, score in st.session_state["report_keywords"]:
            pdf.cell(200, 10, txt=f"- {word} (score: {score:.2f})", ln=True)

        pdf.cell(200, 10, txt="🤖 AI Recommendations:", ln=True)
        for rec in st.session_state["report_ai"]:
            pdf.cell(200, 10, txt=f"- {rec}", ln=True)

        pdf.cell(200, 10, txt="🛠 Guided Steps:", ln=True)
        for step in st.session_state["report_steps"]:
            for line in step.split("\n"):
                pdf.cell(200, 10, txt=line.strip(), ln=True)

        pdf_output = "final_report.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_output}">📥 Download Report PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("ℹ️ Run the Note Analyzer first to generate a report.")

