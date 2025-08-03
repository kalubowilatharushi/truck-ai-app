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

# ---------- Load Users (For Login)
def load_users():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password):
    df = load_users()
    df = df.append({"username": username, "password": hash_password(password)}, ignore_index=True)
    df.to_csv("users.csv", index=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    df = load_users()
    hashed = hash_password(password)
    return ((df['username'] == username) & (df['password'] == hashed)).any()
