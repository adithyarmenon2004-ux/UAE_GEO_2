import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("🚀 Fixed Travel Analytics Dashboard")

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Encode consistently
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    X = df_enc.drop("Likelihood", axis=1)
    y = df_enc["Likelihood"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    st.subheader("Model Trained Successfully")

    st.subheader("🧪 New Customer Prediction")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(col, value=0)

    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])

        # Fix: match training columns exactly
        user_df = user_df.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(user_df)
        st.success(f"Prediction: {prediction}")
