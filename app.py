import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import base64
import io

st.set_page_config(page_title="Truck AI Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")

data = load_data()

# Model training
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}
predictions = model.predict(X)
data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

# Navigation
tab = st.sidebar.radio("📂 Navigate", ["Overview", "Notes Analyzer", "Reports"])

# Overview Tab
if tab == "Overview":
    st.title("🚚 Truck Health Overview")
    st.dataframe(data)

    label_series = pd.Series([labels[p] for p in predictions])
    emoji_counts = label_series.map(emoji_map).value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low Risk", emoji_counts.get('🟢 Low', 0))
    col2.metric("🟡 Medium Risk", emoji_counts.get('🟡 Medium', 0))
    col3.metric("🔴 High Risk", emoji_counts.get('🔴 High', 0))

# Notes Analyzer Tab
elif tab == "Notes Analyzer":
    st.title("🛠️ Mechanic Notes Analyzer")
    st.write("Paste comments to see keywords and get AI suggestions.")

    text_input = st.text_area("📝 Mechanic Comment", placeholder="E.g., Engine overheating, oil pressure low...")

    if st.button("Analyze"):
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

            st.success("✅ Top Keywords Detected:")
            for word, score in sorted_keywords:
                st.markdown(f"- `{word}` (score: {score:.2f})")

            st.markdown("### 💡 AI Suggestions:")
            suggestions = []
            comment = text_input.lower()
            if "oil" in comment:
                suggestions.append("Check for oil leaks or clogged oil filters.")
            if "overheat" in comment or "coolant" in comment:
                suggestions.append("Inspect radiator and coolant level.")
            if "rpm" in comment or "idle" in comment:
                suggestions.append("Throttle body or sensor may need calibration.")
            if "vibration" in comment:
                suggestions.append("Inspect engine mounts or wheels.")
            if not suggestions:
                suggestions.append("No urgent issues detected. Further inspection may be needed.")

            for tip in suggestions:
                st.markdown(f"🔧 {tip}")

# Reports Tab
elif tab == "Reports":
    st.title("🧾 Maintenance Report Generator")
    st.write("Download a full report of the latest truck health summary.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total = len(data)
    low = data['Failure Risk'].value_counts().get('🟢 Low', 0)
    med = data['Failure Risk'].value_counts().get('🟡 Medium', 0)
    high = data['Failure Risk'].value_counts().get('🔴 High', 0)

    if st.button("📄 Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Truck AI Health Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Generated on: {now}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Total Records: {total}", ln=True)
        pdf.cell(200, 10, txt=f"Low Risk: {low} | Medium Risk: {med} | High Risk: {high}", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="This report summarizes the latest prediction results.", ln=True)

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        b64 = base64.b64encode(pdf_output.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="truck_ai_report.pdf">📥 Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
