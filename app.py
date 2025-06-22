import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------
# Page configuration
# -------------------
st.set_page_config(page_title="Isuzu 4JJ1 AI Maintenance System", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚚 Isuzu 4JJ1 - AI Maintenance Dashboard</h1>", unsafe_allow_html=True)

# -------------------
# Load inbuilt dataset
# -------------------
@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")

data = load_data()

# -------------------
# Train Model
# -------------------
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}

# -------------------
# Tab Layout
# -------------------
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# -------------------
# TAB 1: Dashboard
# -------------------
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

# -------------------
# TAB 2: Note Analyzer with AI Suggestions
# -------------------
with tab2:
    st.markdown("### 🛠️ Mechanic Notes Analyzer")
    st.write("Enter mechanic comments below to detect key issues and get AI suggestions.")

    text_input = st.text_area("📝 Mechanic Comment", placeholder="E.g., Oil leak detected near the filter. RPM drops at idle...", height=150)
    submit = st.button("Analyze Comments")

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

            # Simple AI-based suggestions
            st.markdown("### 🤖 AI-Based Recommendations:")
            for word, _ in sorted_keywords:
                if "oil" in word:
                    st.markdown("- 🔧 Check oil filter and oil level.")
                elif "coolant" in word:
                    st.markdown("- 💧 Inspect the coolant system for leaks.")
                elif "rpm" in word or "idle" in word:
                    st.markdown("- ⚙️ Inspect throttle body and idle control valve.")
                elif "engine" in word:
                    st.markdown("- 🛠️ Perform full engine diagnostic.")
                elif "leak" in word:
                    st.markdown("- 🚿 Trace fluid leaks using pressure test.")

# -------------------
# TAB 3: Downloadable Report (Placeholder)
# -------------------
with tab3:
    st.markdown("### 📄 Download Maintenance Report")
    st.write("This section will generate a PDF report of truck diagnostics and NLP suggestions.")
    st.info("📌 Report download feature coming next. For now, all data is visible in tabs above.")
