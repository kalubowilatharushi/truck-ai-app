import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration
st.set_page_config(page_title="Truck Health Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚚 Truck Health Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload truck data to get AI-powered health prediction</p>", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

# Sample training data
sample_data = {
    'Engine_Temp': [85, 95, 100, 70, 110, 90, 88, 102, 105, 98],
    'Oil_Pressure': [30, 28, 20, 40, 15, 25, 32, 18, 22, 27],
    'RPM': [2000, 2500, 3000, 1800, 3200, 2400, 2100, 2900, 3100, 2600],
    'Mileage': [100000, 120000, 130000, 90000, 140000, 110000, 102000, 135000, 145000, 115000],
    'Failure': [0, 1, 2, 0, 2, 1, 0, 2, 2, 1]
}
df = pd.DataFrame(sample_data)
X = df[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']]
y = df['Failure']
model = RandomForestClassifier()
model.fit(X, y)

labels = {0: 'Low', 1: 'Medium', 2: 'High'}
emoji_map = {
    'Low': '🟢 Low',
    'Medium': '🟡 Medium',
    'High': '🔴 High'
}

# Main app logic
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if {'Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage'}.issubset(data.columns):
        predictions = model.predict(data[['Engine_Temp', 'Oil_Pressure', 'RPM', 'Mileage']])
        data['Failure Risk'] = [emoji_map[labels[p]] for p in predictions]

        st.success("✅ Prediction completed!")
        
        st.subheader("📊 Data Overview")
        st.dataframe(data)

        # Count risk levels
        count_raw = [labels[p] for p in predictions]
        risk_counts = pd.Series(count_raw).map({
            'Low': '🟢 Low',
            'Medium': '🟡 Medium',
            'High': '🔴 High'
        }).value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("🟢 Low Risk", risk_counts.get('🟢 Low', 0))
        col2.metric("🟡 Medium Risk", risk_counts.get('🟡 Medium', 0))
        col3.metric("🔴 High Risk", risk_counts.get('🔴 High', 0))

        st.markdown("---")
        st.subheader("📥 Download Result")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "truck_health_results.csv", "text/csv")

    else:
        st.error("❌ Your CSV must include: Engine_Temp, Oil_Pressure, RPM, Mileage")
else:
    st.info("👆 Please upload a CSV file to begin.")

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
