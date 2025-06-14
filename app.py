import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(page_title="Truck Health Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚚 Truck Health Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered failure risk prediction using 1,000 inbuilt records</p>", unsafe_allow_html=True)

# Load inbuilt dataset
@st.cache_data
def load_data():
    df = pd.read_csv("inbuilt_truck_data.csv")
    return df

data = load_data()

# Train model on inbuilt dataset
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

# Prediction labels
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {
    'Low': '🟢 Low',
    'Medium': '🟡 Medium',
    'High': '🔴 High'
}

# Predict on the same inbuilt data
predictions = model.predict(X)
data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

# Display results
st.success("✅ Prediction completed on 1,000 inbuilt records.")
st.subheader("📊 Prediction Results")
st.dataframe(data)

# Risk counts
label_series = pd.Series([labels[p] for p in predictions])
emoji_counts = label_series.map(emoji_map).value_counts()

col1, col2, col3 = st.columns(3)
col1.metric("🟢 Low Risk", emoji_counts.get('🟢 Low', 0))
col2.metric("🟡 Medium Risk", emoji_counts.get('🟡 Medium', 0))
col3.metric("🔴 High Risk", emoji_counts.get('🔴 High', 0))

# ---------------------------------------
# 🛠️ Mechanic Notes Analyzer (NLP)
# ---------------------------------------
st.markdown("---")
st.subheader("🛠️ Mechanic Notes Analyzer")

text_input = st.text_area("Paste mechanic notes or comments here:")

if st.button("Analyze Comments"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        logs = [text_input]

        vectorizer = TfidfVectorizer(stop_words='english')
        X_text = vectorizer.fit_transform(logs)
        feature_names = vectorizer.get_feature_names_out()
        scores = X_text.toarray().flatten()

        keywords = list(zip(feature_names, scores))
        sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:5]

        st.write("🔍 **Top Keywords Detected:**")
        for word, score in sorted_keywords:
            st.markdown(f"- `{word}` (score: {score:.2f})")
