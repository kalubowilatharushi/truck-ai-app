import streamlit as st
import pandas as pd
import hashlib
import os

# Set up page
st.set_page_config(page_title="DEBUG - Isuzu AI Login")

# ---------- Password Hashing ----------
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------- User Auth ----------
def load_users():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password):
    df = load_users()
    if username in df["username"].values:
        return False
    new_row = pd.DataFrame([{"username": username, "password": hash_pw(password)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("users.csv", index=False)
    return True

def check_login(username, password):
    df = load_users()
    hashed = hash_pw(password)
    
    # DEBUG: Show what's being checked
    st.write("📂 Current users.csv content:")
    st.dataframe(df)

    st.write("🧾 Entered username:", username)
    st.write("🔐 Entered password (plain):", password)
    st.write("🔑 Entered password (hashed):", hashed)

    # Find match
    user_match = df[df["username"] == username]
    if not user_match.empty:
        correct_password = user_match.iloc[0]["password"]
        st.write("🧮 Stored hash in CSV:", correct_password)

        if hashed == correct_password:
            st.success("✅ Hash matches! Login should succeed.")
            return True
        else:
            st.error("❌ Hash mismatch: login failed.")
            return False
    else:
        st.error("❌ Username not found.")
        return False

# ---------- Session ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# ---------- Login/Register UI ----------
if not st.session_state.logged_in:
    st.title("🔐 DEBUG: Login System Test")
    login_tab, register_tab = st.tabs(["🔓 Login", "📝 Register"])

    with login_tab:
        user = st.text_input("Username").strip()
        pw = st.text_input("Password", type="password").strip()
        if st.button("Login"):
            if check_login(user, pw):
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success("🎉 Logged in!")
                st.rerun()
            else:
                st.error("Login failed.")

    with register_tab:
        new_user = st.text_input("New Username").strip()
        new_pw = st.text_input("New Password", type="password").strip()
        if st.button("Register"):
            if new_user and new_pw:
                if save_user(new_user, new_pw):
                    st.success("✅ Registration successful!")
                else:
                    st.warning("⚠ Username already exists.")
            else:
                st.warning("⚠ Please fill both fields.")

    st.stop()

# ---------- Logged In View ----------
st.success(f"Welcome, {st.session_state.user} ✅")
st.write("You are now inside the app. This is the secure area.")
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
