import streamlit as st
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Truck Health AI", layout="wide")

# --- Custom CSS for clean UI ---
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

# --- Tabs for sections ---
tab1, tab2, tab3 = st.tabs(["🛠️ Predict Engine Health", "💬 Analyze Mechanic Notes", "📊 View Engine Data"])

# --- Tab 1: Prediction ---
with tab1:
    st.subheader("Engine Failure Risk")
    
    # Dummy output: Replace with your ML model prediction
    risk_level = "Medium"  # example
    if risk_level == "Low":
        st.success("🟢 Risk Level: Low")
    elif risk_level == "Medium":
        st.warning("🟡 Risk Level: Medium")
    else:
        st.error("🔴 Risk Level: High")
    
    st.metric("Failure Probability", value="42%", delta="-5% since last check")

# --- Tab 2: NLP Analysis ---
with tab2:
    st.subheader("Top Keywords from Mechanic Comments")

    # Dummy keywords: Replace with real NLP results
    keywords = ["oil leak", "sensor error", "engine heat", "delay start"]
    st.write("🔑 **Frequent Issues:**")
    for word in keywords:
        st.write(f"• {word}")

    # Optional textarea for user input
    comment = st.text_area("Paste mechanic comment to analyze:")
    if st.button("Analyze"):
        st.info("🔍 AI Analysis: Possible fault in **engine sensor**.")

# --- Tab 3: Inbuilt Database (Google Sheets) ---
with tab3:
    st.subheader("Truck Performance Data")

    sheet_url = "https://docs.google.com/spreadsheets/d/1ccOkw9pObr27u862s7sg-5qnTj_82KQpL3eayBgtgEg/export?format=csv"
    try:
        df = pd.read_csv(sheet_url)
        st.dataframe(df)
    except:
        st.error("❌ Could not load data from Google Sheets.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Tharushi Kalubowila** | [GitHub Repo](https://github.com/kalubowilatharushi/truck-ai-app)")
import streamlit as st
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Truck Health AI", layout="wide")

# --- Custom CSS for clean UI ---
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

# --- Tabs for sections ---
tab1, tab2, tab3 = st.tabs(["🛠️ Predict Engine Health", "💬 Analyze Mechanic Notes", "📊 View Engine Data"])

# --- Tab 1: Prediction ---
with tab1:
    st.subheader("Engine Failure Risk")
    
    # Dummy output: Replace with your ML model prediction
    risk_level = "Medium"  # example
    if risk_level == "Low":
        st.success("🟢 Risk Level: Low")
    elif risk_level == "Medium":
        st.warning("🟡 Risk Level: Medium")
    else:
        st.error("🔴 Risk Level: High")
    
    st.metric("Failure Probability", value="42%", delta="-5% since last check")

# --- Tab 2: NLP Analysis ---
with tab2:
    st.subheader("Top Keywords from Mechanic Comments")

    # Dummy keywords: Replace with real NLP results
    keywords = ["oil leak", "sensor error", "engine heat", "delay start"]
    st.write("🔑 **Frequent Issues:**")
    for word in keywords:
        st.write(f"• {word}")

    # Optional textarea for user input
    comment = st.text_area("Paste mechanic comment to analyze:")
    if st.button("Analyze"):
        st.info("🔍 AI Analysis: Possible fault in **engine sensor**.")

# --- Tab 3: Inbuilt Database (Google Sheets) ---
with tab3:
    st.subheader("Truck Performance Data")

    sheet_url = "https://docs.google.com/spreadsheets/d/1ccOkw9pObr27u862s7sg-5qnTj_82KQpL3eayBgtgEg/export?format=csv"
    try:
        df = pd.read_csv(sheet_url)
        st.dataframe(df)
    except:
        st.error("❌ Could not load data from Google Sheets.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by **Tharushi Kalubowila** | [GitHub Repo](https://github.com/kalubowilatharushi/truck-ai-app)")
