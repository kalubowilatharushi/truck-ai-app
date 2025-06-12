import streamlit as st

# ✅ Set page config FIRST
st.set_page_config(page_title="Truck Health AI", layout="wide")

import pandas as pd

# --- Custom CSS for modern UI ---
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .card {
            background-color: white;
            padding: 1.5em;
            border-radius: 15px;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            margin-bottom: 1em;
        }
        .header {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .subtext {
            font-size: 1em;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Isuzu_logo.svg/2560px-Isuzu_logo.svg.png", width=180)
st.markdown('<div class="card"><div class="header">🚚 AI-Based Truck Health Prediction App</div><div class="subtext">Monitor Isuzu 4JJ1 engine condition with AI + NLP</div></div>', unsafe_allow_html=True)

# --- Tabs for features ---
tab1, tab2, tab3 = st.tabs(["🛠️ Predict Engine Health", "💬 Analyze Mechanic Notes", "📊 View Engine Data"])

# --- Tab 1: Predict Engine Health ---
with tab1:
    st.subheader("Engine Failure Risk")
    
    # 🔁 Replace with real model prediction here
    risk_level = "Medium"  # Example: Use ML model here

    if risk_level == "Low":
        st.success("🟢 Risk Level: Low - Truck is healthy.")
    elif risk_level == "Medium":
        st.warning("🟡 Risk Level: Medium - Maintenance recommended.")
    else:
        st.error("🔴 Risk Level: High - Immediate attention needed!")

    # Example metric
    st.metric("Failure Probability", value="42%", delta="-5% since last week")

# --- Tab 2: NLP - Analyze Mechanic Notes ---
with tab2:
    st.subheader("Top Keywords from Mechanic Comments")

    # 🔁 Replace with actual NLP output
    keywords = ["oil leak", "sensor error", "engine heat", "delay start"]

    st.markdown("**🔑 Frequent Issues Found:**")
    for kw in keywords:
        st.markdown(f"- {kw}")

    # Optional: Analyze new comment input
    comment = st.text_area("Paste mechanic comment to analyze:")
    if st.button("Analyze"):
        # 🔁 Replace with real NLP model
        st.info("🔍 AI Analysis: Possible issue with engine temperature sensor.")

# --- Tab 3: View Google Sheets Data ---
with tab3:
    st.subheader("Truck Performance Data (from Google Sheets)")

    # Load from Google Sheets
    sheet_url = "https://docs.google.com/spreadsheets/d/1ccOkw9pObr27u862s7sg-5qnTj_82KQpL3eayBgtgEg/export?format=csv"

    try:
        df = pd.read_csv(sheet_url)
        st.dataframe(df)
    except Exception as e:
        st.error("❌ Could not load data from Google Sheets.")
        st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Tharushi Kalubowila**  \n[GitHub Repo](https://github.com/kalubowilatharushi/truck-ai-app) | [Live App](https://truck-ai-app-kjd7oi2patsuru9uttftsl.streamlit.app)")
