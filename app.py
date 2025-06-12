import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Page setup
st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# Custom Header
st.markdown("<h1 style='text-align: center; color: #1f4e79;'>🚛 Isuzu 4JJ1 AI System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI-powered truck maintenance dashboard for prediction and analysis</p>", unsafe_allow_html=True)

# Sidebar Navigation
menu = st.sidebar.radio("📁 Select Section", ["Dashboard Overview", "Truck Health Prediction", "Mechanic Notes Analyzer"])

# Load and prepare data
@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")

data = load_data()

# Train model
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

# Predict
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}
predictions = model.predict(X)
data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

# Section: Dashboard Overview
if menu == "Dashboard Overview":
    st.subheader("📊 Risk Summary Overview")

    label_series = pd.Series([labels[p] for p in predictions])
    emoji_counts = label_series.map(emoji_map).value_counts()

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Low Risk", emoji_counts.get('🟢 Low', 0))
    col2.metric("🟡 Medium Risk", emoji_counts.get('🟡 Medium', 0))
    col3.metric("🔴 High Risk", emoji_counts.get('🔴 High', 0))

    # Bar chart visualization
    st.markdown("### 📈 Risk Level Distribution")
    fig, ax = plt.subplots()
    emoji_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax)
    ax.set_ylabel("Number of Trucks")
    ax.set_title("Truck Health Risk Distribution")
    st.pyplot(fig)

# Section: Health Prediction Table
elif menu == "Truck Health Prediction":
    st.subheader("🔍 Truck Failure Predictions")
    st.dataframe(data)

# Section: Mechanic Notes Analyzer
elif menu == "Mechanic Notes Analyzer":
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
