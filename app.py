import streamlit as st
import pandas as pd
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
st.markdown("""
<h1 style='text-align:center;color:#FF0000;'>🚚 Isuzu 4JJ1 AI Maintenance Dashboard</h1>
""", unsafe_allow_html=True)

# ... [All existing code remains the same above this section] ...

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
