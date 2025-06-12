import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Page config
st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

# --- Header ---
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color:#1f4e79;'>🚛 Isuzu 4JJ1 AI System</h1>
        <p style='color:gray; font-size: 1.1rem;'>Mobile-optimized AI dashboard for truck maintenance</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
menu = st.sidebar.radio("📁 Navigate", ["📊 Dashboard", "🔍 Health Prediction", "🛠️ Notes Analyzer"])

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("inbuilt_truck_data.csv")

data = load_data()

# --- Train Model ---
X = data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = data['Failure']
model = RandomForestClassifier()
model.fit(X, y)

# --- Predict ---
labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {'Low': '🟢 Low', 'Medium': '🟡 Medium', 'High': '🔴 High'}
predictions = model.predict(X)
data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

# --- Dashboard ---
if menu == "📊 Dashboard":
    st.subheader("🚦 Truck Health Summary")

    label_series = pd.Series([labels[p] for p in predictions])
    emoji_counts = label_series.map(emoji_map).value_counts()

    st.markdown(f"""
    <div style='display: flex; gap: 1rem; flex-wrap: wrap; justify-content: space-around;'>
        <div style='flex: 1; min-width: 100px; background: #e8f5e9; padding: 1rem; border-radius: 12px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center;'>
            <h3>🟢 Low Risk</h3>
            <h2>{emoji_counts.get('🟢 Low', 0)}</h2>
        </div>
        <div style='flex: 1; min-width: 100px; background: #fff9c4; padding: 1rem; border-radius: 12px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center;'>
            <h3>🟡 Medium Risk</h3>
            <h2>{emoji_counts.get('🟡 Medium', 0)}</h2>
        </div>
        <div style='flex: 1; min-width: 100px; background: #ffebee; padding: 1rem; border-radius: 12px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-align: center;'>
            <h3>🔴 High Risk</h3>
            <h2>{emoji_counts.get('🔴 High', 0)}</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Risk Level Chart")
    fig, ax = plt.subplots()
    emoji_counts.plot(kind='bar', color=['green', 'orange', 'red'], ax=ax)
    ax.set_ylabel("Number of Trucks")
    ax.set_title("Health Risk Distribution")
    st.pyplot(fig)

# --- Health Prediction ---
elif menu == "🔍 Health Prediction":
    st.subheader("📋 Truck Failure Predictions")
    st.dataframe(data)

# --- NLP Notes Analyzer ---
elif menu == "🛠️ Notes Analyzer":
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
