import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime

# 1. Setup
st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚚 Truck Health Dashboard</h1>", unsafe_allow_html=True)

# 2. Load inbuilt 1000-record dataset
@st.cache_data
def load_data():
    df = pd.read_csv("inbuilt_truck_data.csv")  # You must include this CSV in your repo
    return df

df = load_data()

# 3. Train model
X = df[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = df['Failure']
model = RandomForestClassifier()
model.fit(X, y)
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
df['Predicted_Risk'] = model.predict(X)
df['Risk_Label'] = df['Predicted_Risk'].map(labels)

# 4. Sidebar Navigation
tab = st.sidebar.radio("📂 Select Page", ["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# 5. 🏠 Dashboard Tab
if tab == "🏠 Dashboard":
    st.subheader("📊 Fleet Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Low Risk", str(sum(df['Risk_Label'] == 'Low')))
    col2.metric("Medium Risk", str(sum(df['Risk_Label'] == 'Medium')))
    col3.metric("High Risk", str(sum(df['Risk_Label'] == 'High')))

    st.markdown("---")
    st.subheader("📈 Risk Level Distribution")

    risk_counts = df['Risk_Label'].value_counts()
    fig1, ax1 = plt.subplots()
    risk_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_title("Predicted Failure Risk Levels")
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("🥧 Risk Pie Chart")

    fig2, ax2 = plt.subplots()
    ax2.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🚛 Recent Activity")
    st.markdown("""
    - ✅ Model trained on 1000 real entries  
    - 📤 Dashboard synced with prediction output  
    - 📥 Fleet insights updated live  
    """)

# 6. 🛠️ Note Analyzer
elif tab == "🛠️ Note Analyzer":
    st.subheader("🛠️ Mechanic Note Analyzer")

    note = st.text_area("✏️ Enter a mechanic comment:")
    if st.button("Analyze"):
        if note.strip() == "":
            st.warning("⚠️ Please enter a comment before submitting.")
        else:
            # NLP
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform([note])
            keywords = vectorizer.get_feature_names_out()
            scores = tfidf.toarray().flatten()
            top_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:5]

            st.success("✅ Top Keywords:")
            for word, score in top_keywords:
                st.markdown(f"- `{word}` (Score: {score:.2f})")

            # AI Suggestions (basic)
            ai_suggestions = []
            if "oil" in note:
                ai_suggestions.append("Check oil pressure and filter system.")
            if "coolant" in note or "temperature" in note:
                ai_suggestions.append("Inspect cooling system and radiator.")

            st.subheader("🤖 AI Recommendations")
            if ai_suggestions:
                for rec in ai_suggestions:
                    st.markdown(f"- {rec}")
            else:
                st.info("No recommendations found.")

            # Save for report
            st.session_state['keywords'] = top_keywords
            st.session_state['recommendations'] = ai_suggestions

# 7. 📄 Report Tab
elif tab == "📄 Report":
    st.subheader("📄 Generate Maintenance Report")

    if 'keywords' not in st.session_state:
        st.info("Analyze a comment first in the Note Analyzer tab.")
    else:
        if st.button("📥 Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Truck Maintenance Report", ln=True, align="C")
            pdf.cell(200, 10, txt=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Top Keywords", ln=True)
            pdf.set_font("Arial", size=12)
            for kw, sc in st.session_state['keywords']:
                pdf.cell(200, 10, txt=f"- {kw} (score: {sc:.2f})", ln=True)

            pdf.ln(5)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="AI Suggestions", ln=True)
            pdf.set_font("Arial", size=12)
            for rec in st.session_state['recommendations']:
                pdf.multi_cell(0, 10, txt=f"- {rec}")

            file_path = "maintenance_report.pdf"
            pdf.output(file_path)
            with open(file_path, "rb") as f:
                st.download_button("📤 Download PDF", f, file_name="maintenance_report.pdf", mime="application/pdf")
