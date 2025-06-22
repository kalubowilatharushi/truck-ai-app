import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="centered")
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🚚 Isuzu 4JJ1 - Mobile AI Maintenance Dashboard</h2>", unsafe_allow_html=True)

# Load inbuilt dataset
@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")

data = load_data()

# Train Model
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}

# Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🛠️ Notes", "📄 Report"])

# Tab 1 - Dashboard
with tab1:
    st.markdown("### 📊 Truck Health Overview")

    predictions = model.predict(X)
    data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]
    st.dataframe(data, use_container_width=True)

    label_series = pd.Series([labels[p] for p in predictions])
    emoji_counts = label_series.map(emoji_map).value_counts()

    st.markdown("### 🚦 Failure Risk Summary")
    st.metric("🟢 Low", emoji_counts.get('🟢 Low', 0))
    st.metric("🟡 Medium", emoji_counts.get('🟡 Medium', 0))
    st.metric("🔴 High", emoji_counts.get('🔴 High', 0))

# Tab 2 - Note Analyzer
with tab2:
    st.markdown("### 🛠️ Mechanic Note Analyzer")
    text_input = st.text_area("✍️ Enter mechanic comments", height=150, placeholder="e.g. Oil leak near the radiator, RPM unstable...")

    analyze = st.button("📎 Analyze Note")
    keywords_list = []
    ai_suggestions = []

    if analyze:
        if text_input.strip() == "":
            st.warning("⚠️ Please enter a comment before submitting.")
        else:
            logs = [text_input]
            vectorizer = TfidfVectorizer(stop_words='english')
            X_text = vectorizer.fit_transform(logs)
            feature_names = vectorizer.get_feature_names_out()
            scores = X_text.toarray().flatten()

            keywords = list(zip(feature_names, scores))
            sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:5]

            st.success("✅ Top Keywords:")
            for word, score in sorted_keywords:
                st.markdown(f"- `{word}` (score: {score:.2f})")
                keywords_list.append(word)

            st.markdown("### 🤖 AI-Based Suggestions")
            for word, _ in sorted_keywords:
                if "oil" in word:
                    sug = "- 🔧 Check oil filter and oil level."
                elif "coolant" in word:
                    sug = "- 💧 Inspect coolant system."
                elif "rpm" in word or "idle" in word:
                    sug = "- ⚙️ Inspect throttle body & idle control valve."
                elif "engine" in word:
                    sug = "- 🛠️ Run engine diagnostics."
                elif "leak" in word:
                    sug = "- 🚿 Pressure test for leaks."
                else:
                    sug = "- ⚙️ Perform standard inspection."
                if sug not in ai_suggestions:
                    ai_suggestions.append(sug)
                    st.markdown(sug)

# Tab 3 - Report Download
with tab3:
    st.markdown("### 📄 Generate PDF Report")

    def generate_pdf(keywords_list, ai_suggestions):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Isuzu 4JJ1 - Maintenance Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 10, txt="Top Keywords:", ln=True)
        for kw in keywords_list:
            pdf.cell(200, 10, txt=f"- {kw}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt="AI Suggestions:", ln=True)
        for sug in ai_suggestions:
            pdf.multi_cell(200, 10, txt=sug)
        return pdf

    if st.button("📥 Generate PDF Report"):
        if not keywords_list or not ai_suggestions:
            st.warning("⚠️ Please analyze notes in the Notes tab first.")
        else:
            pdf = generate_pdf(keywords_list, ai_suggestions)
            pdf_path = "/mnt/data/mobile_maintenance_report.pdf"
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download PDF", f, file_name="maintenance_report.pdf")
