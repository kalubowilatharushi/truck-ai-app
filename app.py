import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# ---------- Load Users (For Login)
def load_users():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password):
    df = load_users()
    df = df.append({"username": username, "password": hash_password(password)}, ignore_index=True)
    df.to_csv("users.csv", index=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    df = load_users()
    hashed = hash_password(password)
    return ((df['username'] == username) & (df['password'] == hashed)).any()

# ---------- Session
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = ""

# ---------- Login / Register UI
if not st.session_state.logged_in:
    st.title("🔐 Truck AI Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("✅ Login successful. Please refresh the page manually.")
                st.stop()
            else:
                st.error("❌ Invalid credentials.")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            save_user(new_user, new_pass)
            st.success("✅ Registration successful. Please log in.")
    st.stop()

# ---------- Load Inbuilt Dataset (1000 samples)
@st.cache(allow_output_mutation=True)
def load_data():
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        'Engine_Temp': rng.integers(70, 110, 1000),
        'Oil_Pressure': rng.integers(15, 35, 1000),
        'RPM': rng.integers(1800, 3500, 1000),
        'Mileage': rng.integers(90000, 180000, 1000)
    })
    return df

data = load_data()

# ---------- Failure Classification
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

# ---------- Detect Engine Issues
def detect_issue(row):
    issues = []
    if row['Engine_Temp'] > 100:
        issues.append("Cooling System")
    if row['Oil_Pressure'] < 20:
        issues.append("Oil System")
    if row['RPM'] > 3000 and row['Engine_Temp'] > 95:
        issues.append("Spark Plug / Vibrations")
    if row['Mileage'] > 140000:
        issues.append("Aging Components")
    return ", ".join(issues) if issues else "Normal"

data['Possible_Issue'] = data.apply(detect_issue, axis=1)

# ---------- Train Model
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

# ---------- Navigation
page = st.sidebar.radio("Navigate", ["Dashboard", "Note Analyzer", "Report"])

# ---------- Dashboard Page
if page == "Dashboard":
    st.title("🚛 Truck Health Prediction Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trucks", len(data))
    col2.metric("High Risk", (data['Failure'] == 2).sum())
    col3.metric("Issues Found", (data['Possible_Issue'] != "Normal").sum())

    # Pie Chart
    st.subheader("🧩 Failure Distribution")
    pie_data = data['Possible_Issue'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    ax1.axis('equal')
    st.pyplot(fig1)

    # Bar Chart
    st.subheader("🔧 Failure Frequency")
    fig2, ax2 = plt.subplots()
    pie_data.plot(kind='bar', color='tomato', ax=ax2)
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    st.subheader("📋 Detailed Truck Data")
    st.dataframe(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage', 'Risk_Level', 'Possible_Issue']])

# ---------- Note Analyzer
elif page == "Note Analyzer":
    st.title("🛠️ Mechanic Note Analyzer")
    text = st.text_area("Paste mechanic notes:")
    if st.button("Analyze"):
        if not text.strip():
            st.warning("⚠️ Please enter mechanic notes to analyze.")
        else:
            tfidf = TfidfVectorizer(stop_words='english')
            matrix = tfidf.fit_transform([text])
            keywords = tfidf.get_feature_names_out()
            scores = matrix.toarray()[0]
            top_keywords = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)[:5]

            if not top_keywords:
                st.warning("❌ No meaningful keywords found.")
            else:
                st.write("**Top Keywords:**")
                for word, score in top_keywords:
                    st.write(f"- {word} ({score:.2f})")

                recommendations = []
                for word, _ in top_keywords:
                    if "oil" in word:
                        recommendations.append("🔧 Check oil filter and pressure.")
                    elif "engine" in word:
                        recommendations.append("🚨 Inspect engine block and coolant system.")
                    elif "coolant" in word:
                        recommendations.append("💧 Refill or replace coolant.")
                    elif "vibration" in word or "noise" in word:
                        recommendations.append("⚙️ Check spark plugs and mounts.")

                if recommendations:
                    st.markdown("---")
                    st.write("### 🤖 AI Recommendations:")
                    for rec in recommendations:
                        st.write(rec)

                st.session_state['keywords'] = top_keywords
                st.session_state['notes'] = text
                st.session_state['recs'] = recommendations

# ---------- Report Generator
elif page == "Report":
    st.title("📄 Generate PDF Report")
    if 'keywords' not in st.session_state:
        st.info("Please analyze mechanic notes first.")
    else:
        if st.button("📥 Download PDF"):
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
                    pdf.multi_cell(0, 10, rec)
            filename = "truck_maintenance_report.pdf"
            pdf.output(filename.encode('latin-1', 'replace'))
            with open(filename, "rb") as f:
                st.download_button("Download Report", f, file_name=filename, mime="application/pdf")
