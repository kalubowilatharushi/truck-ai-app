
import hashlib
import os
import re
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

def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")
st.markdown("<h1 style='text-align:center;color:#FF0000;'>🚚 Isuzu 4JJ1 AI Maintenance Dashboard</h1>", unsafe_allow_html=True)

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "steps" not in st.session_state:
    st.session_state.steps = []

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

                st.session_state.keywords = [kw[0] for kw in sorted_keywords]
                st.session_state.recommendations.clear()
                st.session_state.steps.clear()

                st.success("✅ Top Keywords:")
                for word, score in sorted_keywords:
                    st.markdown(f"- `{word}` (score: {score:.2f})")

                st.markdown("### 🤖 AI-Based Recommendations and Steps")
                for word, _ in sorted_keywords:
                    if "oil" in word:
                        st.session_state.recommendations.append("🔧 Check oil filter and oil level.")
                        st.session_state.steps.append("1. Locate oil filter.\\n2. Inspect for leaks.\\n3. Replace if clogged.")
                    elif "coolant" in word:
                        st.session_state.recommendations.append("💧 Inspect coolant system.")
                        st.session_state.steps.append("1. Check coolant level.\\n2. Look for leaks.\\n3. Pressure test radiator.")
                    elif "rpm" in word or "idle" in word:
                        st.session_state.recommendations.append("⚙️ RPM irregular – check sensors.")
                        st.session_state.steps.append("1. Scan idle speed.\\n2. Inspect throttle body.\\n3. Calibrate if needed.")
                    elif "engine" in word:
                        st.session_state.recommendations.append("🛠️ Run engine diagnostics.")
                        st.session_state.steps.append("1. Connect OBD scanner.\\n2. Analyze fault codes.\\n3. Schedule deep inspection.")
                    elif "leak" in word:
                        st.session_state.recommendations.append("🚿 Pressure test for leaks.")
                        st.session_state.steps.append("1. Identify fluid source.\\n2. Use dye tracer.\\n3. Seal or replace part.")
                    else:
                        st.session_state.recommendations.append("⚙️ Perform general inspection.")
                        st.session_state.steps.append("1. Review truck logs.\\n2. Visual check.\\n3. Record anomalies.")

                for rec, step in zip(st.session_state.recommendations, st.session_state.steps):
                    st.markdown(f"**{rec}**\n\n{step}")

    with tab4:
        st.markdown("### 📄 Preview & Download Report")

        if st.session_state.keywords:
            st.write("#### ✅ Summary Preview")
            st.markdown("**Top Keywords:**")
            for kw in st.session_state.keywords:
                st.markdown(f"- {kw}")

            st.markdown("**AI Recommendations with Steps:**")
            for rec, step in zip(st.session_state.recommendations, st.session_state.steps):
                st.markdown(f"**{rec}**\n\n{step}")

            def generate_pdf(keywords_list, ai_recs, steps):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Isuzu 4JJ1 - Maintenance Report", ln=True, align='C')
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=11)
                pdf.cell(200, 10, txt="Top Keywords:", ln=True)
                for kw in keywords_list:
                    pdf.cell(200, 10, txt=f"- {remove_emojis(kw)}", ln=True)
                pdf.ln(5)
                pdf.cell(200, 10, txt="AI Recommendations:", ln=True)
                for rec, step in zip(ai_recs, steps):
                    pdf.multi_cell(200, 10, txt=remove_emojis(f"{rec}\n{step}"))
                return pdf

            if st.button("📥 Download Report PDF"):
                pdf = generate_pdf(st.session_state.keywords, st.session_state.recommendations, st.session_state.steps)
                path = "maintenance_report.pdf"
                pdf.output(path)
                with open(path, "rb") as f:
                    st.download_button("📄 Download PDF", f, file_name="maintenance_report.pdf")
        else:
            st.info("📝 Please analyze a mechanic comment first from the 'Notes' tab.")
