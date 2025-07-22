
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")

st.markdown("<h1 style='text-align: center;'>🚚 Isuzu 4JJ1 AI Truck Health System</h1>", unsafe_allow_html=True)
tabs = st.tabs(["🏠 Dashboard", "🛠️ Note Analyzer", "📄 Report"])

# ========== LOAD CSV DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("inbuilt_truck_data.csv")

    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = "Low"
    if "Possible_Issue" not in df.columns:
        df["Possible_Issue"] = "Normal"

    return df

df = load_data()

# ========== DASHBOARD TAB ==========
with tabs[0]:
    st.subheader("📊 Truck Health Overview")
    st.dataframe(df)

    col1, col2 = st.columns(2)

    # Risk Level Pie Chart
    with col1:
        risk_counts = df["Risk_Level"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Failure Risk Distribution")
        st.pyplot(fig1)

    # Simplified Issue Bar Chart
    with col2:
        issues_series = df["Possible_Issue"].dropna().astype(str)
        flat_issues = issues_series.str.split(", ").explode().str.strip()
        top_issues = flat_issues.value_counts().head(5)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=top_issues.values, y=top_issues.index, ax=ax2)
        ax2.set_title("Top 5 Failure Causes")
        ax2.set_xlabel("Count")
        st.pyplot(fig2)
