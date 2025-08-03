import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

st.set_page_config(page_title="Isuzu 4JJ1 AI System", layout="wide")
st.title("Isuzu 4JJ1 AI System")

# ---------- User Authentication ----------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password):
    df = load_users()
    new_row = pd.DataFrame([{"username": username, "password": hash_password(password)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("users.csv", index=False)

def check_login(username, password):
    df = load_users()
    hashed = hash_password(password)
    return ((df['username'] == username) & (df['password'] == hashed)).any()

# ---------- Session Setup ----------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = ""

# ---------- Login/Register Interface ----------
if not st.session_state.logged_in:
    st.title("🔐 Truck AI Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.success("Login successful")
                st.session_state.logged_in = True
                st.session_state.user = username
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Register"):
            save_user(new_user, new_pass)
            st.success("User registered! Please log in.")

# ---------- Main App Interface ----------
else:
    st.success(f"Welcome, {st.session_state.user}! 🚚")
    st.markdown("### 🔍 Truck Diagnostic System")

    # Example Inputs
    engine_noise = st.text_input("Engine Noise Description")
    rpm = st.slider("Engine RPM", 500, 4000, 1500)
    temperature = st.slider("Engine Temperature (°C)", 40, 120, 90)

    # Simulated TF-IDF and model logic
    if st.button("Diagnose"):
        if not engine_noise:
            st.warning("Please enter engine noise description.")
        else:
            # Example vectorizer and model (mock-up)
            st.info("Running diagnosis...")

            tfidf = TfidfVectorizer()
            X = tfidf.fit_transform([engine_noise])
            features = pd.DataFrame(X.toarray())

            features["rpm"] = rpm
            features["temp"] = temperature

            # Fake model (random forest classifier stub)
            model = RandomForestClassifier()
            try:
                model.fit(features, [0])  # dummy fit
                prediction = model.predict(features)[0]
                st.success(f"Prediction: {'Fault Detected' if prediction else 'Normal'}")
            except Exception as e:
                st.error(f"Model error: {e}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = ""
        st.experimental_rerun()
