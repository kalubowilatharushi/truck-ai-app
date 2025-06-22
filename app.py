import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Isuzu 4JJ1 AI Maintenance System", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚚 Isuzu 4JJ1 - AI Maintenance Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")

data = load_data()

X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}

# Tab layout
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

with tab1:
    st.markdown("### 🔍 Health Prediction Based on Inbuilt Data")
    predictions = model.predict(X)
    data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]
    st.dataframe(data)

    label_series = pd.Series([labels[p] for p in predictions])
    emoji_counts = label_series.map(emoji_map).value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low Risk", emoji_counts.get('🟢 Low', 0))
    col2.metric("🟡 Medium Risk", emoji_counts.get('🟡 Medium', 0))
    col3.metric("🔴 High Risk", emoji_counts.get('🔴 High', 0))

with tab2:
    st.markdown("### 🛠️ Mechanic Notes Analyzer")
    st.write("Enter mechanic comments below to detect key issues and get AI suggestions.")

    text_input = st.text_area("📝 Mechanic Comment", placeholder="E.g., Oil leak detected near the filter. RPM drops at idle...", height=150)
    submit = st.button("Analyze Comments")

    keywords_list = []
    ai_suggestions = []

    if submit:
        if text_input.strip() == "":
            st.error("❌ Please enter a comment before submitting.")
        else:
            logs = [text_input]
            vectorizer = TfidfVectorizer(stop_words='english')
            X_text = vectorizer.fit_transform(logs)
            feature_names = vectorizer.get_feature_names_out()
            scores = X_text.toarray().flatten()

            keywords = list(zip(feature_names, scores))
            sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:5]

            st.success("✅ Top Keywords Detected:")
            for word, score in sorted_keywords:
                st.markdown(f"- `{word}` (score: {score:.2f})")
                keywords_list.append(word)

            st.markdown("### 🤖 AI-Based Recommendations:")
            for word, _ in sorted_keywords:
                if "oil" in word:
                    suggestion = "- 🔧 Check oil filter and oil level."
                elif "coolant" in word:
                    suggestion = "- 💧 Inspect the coolant system for leaks."
                elif "rpm" in word or "idle" in word:
                    suggestion = "- ⚙️ Inspect throttle body and idle control valve."
                elif "engine" in word:
                    suggestion = "- 🛠️ Perform full engine diagnostic."
                elif "leak" in word:
                    suggestion = "- 🚿 Trace fluid leaks using pressure test."
                else:
                    suggestion = "- ⚙️ General inspection recommended."

                if suggestion not in ai_suggestions:
                    ai_suggestions.append(suggestion)
                    st.markdown(suggestion)

with tab3:
    st.markdown("### 📄 Download Maintenance Report")
    st.write("Click below to generate and download a PDF report based on the current analysis.")

    def generate_pdf(keywords_list, ai_suggestions):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Isuzu 4JJ1 - AI Maintenance Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 10, txt="Top Keywords from Mechanic Notes:", ln=True)
        for kw in keywords_list:
            pdf.cell(200, 10, txt=f"- {kw}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt="AI-Based Recommendations:", ln=True)
        for sug in ai_suggestions:
            pdf.multi_cell(200, 10, txt=sug)
        return pdf

    if st.button("📥 Download PDF Report"):
        if not keywords_list or not ai_suggestions:
            st.warning("⚠️ Please analyze a mechanic comment first in the 'Note Analyzer' tab.")
        else:
            pdf = generate_pdf(keywords_list, ai_suggestions)
            pdf.output("/mnt/data/maintenance_report.pdf")
            with open("/mnt/data/maintenance_report.pdf", "rb") as f:
                st.download_button("📄 Click to Download PDF", f, file_name="maintenance_report.pdf")
