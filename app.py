import streamlit as st
import pandas as pd
import hashlib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
from datetime import datetime
import plotly.express as px

USER_FILE = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_user(username, password_hash):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([[username, password_hash]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_FILE, index=False)
    return True

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")
st.markdown("<h1 style='text-align:center;color:#FF0000;'>🚚 Isuzu 4JJ1 AI Maintenance Dashboard</h1>", unsafe_allow_html=True)

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = ""

if choice == "Register":
    st.subheader("🔐 Register")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if new_pass != confirm_pass:
            st.warning("Passwords do not match.")
        elif new_user.strip() == "" or new_pass.strip() == "":
            st.warning("Please enter valid credentials.")
        else:
            hashed = hash_password(new_pass)
            if save_user(new_user, hashed):
                st.success("Account created! You can now log in.")
            else:
                st.error("Username already exists.")

elif choice == "Login":
    st.subheader("🔐 Login")
    user = st.text_input("Username")
    passwd = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_users()
        if user in users["username"].values:
            hashed = hash_password(passwd)
            if users[users["username"] == user]["password"].values[0] == hashed:
                st.session_state.auth = True
                st.session_state.user = user
                st.success(f"✅ Welcome, {user}!")
            else:
                st.error("Incorrect password.")
        else:
            st.error("User not found.")

if st.session_state.auth:
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
    predictions = model.predict(X)
    data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]
    label_series = pd.Series([labels[p] for p in predictions])
    emoji_counts = label_series.map(emoji_map).value_counts()

    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 Visualize", "🛠️ Notes", "📄 Report"])

    with tab1:
        st.markdown("### 📊 Truck Health Overview")
        st.dataframe(data, use_container_width=True)
        st.metric("🟢 Low", emoji_counts.get('🟢 Low', 0))
        st.metric("🟡 Medium", emoji_counts.get('🟡 Medium', 0))
        st.metric("🔴 High", emoji_counts.get('🔴 High', 0))

    with tab2:
        st.markdown("### 📈 Failure Risk Charts")
        bar_fig = px.bar(x=emoji_counts.index, y=emoji_counts.values, color=emoji_counts.index,
                         labels={"x": "Risk Level", "y": "Truck Count"},
                         title="Failure Risk Distribution")
        st.plotly_chart(bar_fig, use_container_width=True)

        pie_fig = px.pie(names=emoji_counts.index, values=emoji_counts.values,
                         title="Risk Proportion", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(pie_fig, use_container_width=True)

    with tab3:
        st.markdown("### 🛠️ Mechanic Note Analyzer")
        text_input = st.text_area("✍️ Enter mechanic comments", height=150)
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

    with tab4:
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
                pdf_path = "maintenance_report.pdf"
                pdf.output(pdf_path)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF", f, file_name="maintenance_report.pdf")
