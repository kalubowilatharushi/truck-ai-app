import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# ----------------- User Auth Functions -----------------
def load_users():
    if os.path.exists("users.csv"):
        return pd.read_csv("users.csv")
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    users = load_users()
    if username in users['username'].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)
    return True

def check_login(username, password):
    users = load_users()
    return not users[(users["username"] == username) & (users["password"] == password)].empty

# ----------------- Load Dataset -----------------
@st.cache(allow_output_mutation=True)
def load_data():
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        'Engine_Temp': rng.integers(70, 110, 1000),
        'Oil_Pressure': rng.integers(15, 35, 1000),
        'RPM': rng.integers(1800, 3500, 1000),
        'Mileage': rng.integers(90000, 180000, 1000)
    })
    return df.copy()

data = load_data()

# ----------------- AI Logic -----------------
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

X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

# ----------------- Login / Register -----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    with tab1:
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(user, pw):
                st.success("✅ Login successful!")
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
    with tab2:
        new_user = st.text_input("Create Username")
        new_pw = st.text_input("Create Password", type="password")
        if st.button("Register"):
            if save_user(new_user, new_pw):
                st.success("✅ Account created. Please log in.")
            else:
                st.warning("⚠️ Username already exists.")
    st.stop()

# ----------------- Navigation -----------------
page = st.sidebar.radio("Navigate", ["Dashboard", "Note Analyzer", "Report"])

# ----------------- Dashboard -----------------
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

# ----------------- Note Analyzer -----------------
elif page == "Note Analyzer":
    st.title("🛠️ Mechanic Note Analyzer")
    text = st.text_area("Paste mechanic notes:")
    if st.button("Analyze"):
        if not text.strip():
            st.warning("⚠️ Please enter mechanic notes to analyze.")
        else:
            tfidf = TfidfVectorizer(stop_words='english')
            matrix = tfidf.fit_transform([text])
            if matrix.shape[1] == 0:
                st.error("❌ Could not extract meaningful keywords. Please enter valid notes.")
            else:
                keywords = tfidf.get_feature_names_out()
                scores = matrix.toarray()[0]
                top_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:5]
                st.write("**Top Keywords:**")
                for word, score in top_keywords:
                    st.write(f"- {word} ({score:.2f})")

                # AI Recommendations
                recommendations = []
                for word, _ in top_keywords:
                    if "oil" in word:
                        recommendations.append("🔧 Check oil filter and pressure levels.")
                    elif "engine" in word:
                        recommendations.append("🚨 Inspect engine block and coolant system.")
                    elif "coolant" in word:
                        recommendations.append("💧 Top-up or replace coolant.")
                    elif "knocking" in word or "vibration" in word:
                        recommendations.append("⚙️ Investigate spark plugs and engine mounts.")
                if recommendations:
                    st.markdown("---")
                    st.write("### 🤖 AI Suggestions:")
                    for tip in recommendations:
                        st.write(tip)
                else:
                    st.info("🤖 No specific AI recommendations found.")

                st.session_state['keywords'] = top_keywords
                st.session_state['notes'] = text
                st.session_state['recs'] = recommendations

# ----------------- Report -----------------
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
            if 'recs' in st.session_state:
                pdf.ln(5)
                pdf.cell(200, 10, "AI Recommendations:", ln=True)
                for rec in st.session_state['recs']:
                    # Remove emoji before writing to PDF
                    clean_text = ''.join(c for c in rec if ord(c) < 128)
                    pdf.multi_cell(0, 10, clean_text)
            filename = "truck_maintenance_report.pdf"
            pdf.output(filename)
            with open(filename, "rb") as f:
                st.download_button("⬇️ Download Report", f, file_name=filename, mime="application/pdf")
