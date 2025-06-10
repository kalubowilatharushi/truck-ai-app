import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------- Page Setup -------------------
st.set_page_config(page_title="Truck Health Dashboard", layout="centered")
st.markdown("""<style>
body {
    background-color: #f5f7fa;
    color: #333333;
}
[data-testid="stMetricValue"] {
    font-size: 24px;
}
</style>""", unsafe_allow_html=True)

st.markdown("## 🚚 AI-Based Truck Health Prediction")
st.caption("Welcome! This tool helps predict truck engine condition and analyze mechanic notes using built-in AI.")

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("inbuilt_truck_data.csv")
    return df

data = load_data()

# ------------------- Train Model -------------------
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}

# Predict
predictions = model.predict(X)
data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

# ------------------- Show Prediction Table -------------------
st.markdown("### 📊 Prediction Overview")
st.success("AI has analyzed 1,000 truck records.")

with st.expander("🔍 View prediction table"):
    st.dataframe(data)

# ------------------- Metrics -------------------
label_series = pd.Series([labels[p] for p in predictions])
emoji_counts = label_series.map(emoji_map).value_counts()

col1, col2, col3 = st.columns(3)
col1.metric("🟢 Low Risk", emoji_counts.get('🟢 Low', 0))
col2.metric("🟡 Medium Risk", emoji_counts.get('🟡 Medium', 0))
col3.metric("🔴 High Risk", emoji_counts.get('🔴 High', 0))

# ------------------- Mechanic Notes Analyzer -------------------
st.markdown("---")
st.markdown("### 🛠️ Mechanic Notes Analyzer")
st.caption("Paste any notes or comments from vehicle inspections to identify key issues.")

text_input = st.text_area("📄 Enter mechanic notes")

if st.button("🔎 Analyze Text"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter mechanic notes to analyze.")
    else:
        logs = [text_input]
        vectorizer = TfidfVectorizer(stop_words='english')
        X_text = vectorizer.fit_transform(logs)
        feature_names = vectorizer.get_feature_names_out()
        scores = X_text.toarray().flatten()

        keywords = list(zip(feature_names, scores))
        sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:5]

        st.success("✅ Top keywords detected:")
        for word, score in sorted_keywords:
            st.markdown(f"- `{word}` (Score: {score:.2f})")