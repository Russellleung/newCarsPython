import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np

# Streamlit App Title
st.title("📊 Advanced Auto Analytics Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Show Data Preview
    st.subheader("📌 Data Preview")
    st.dataframe(df.head())
    
    # Show Basic Stats
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())
    
    # Correlation Analysis
    st.subheader("🔗 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['number']).columns
    fig, ax = plt.subplots()
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Anomaly Detection using Isolation Forest
    st.subheader("⚠️ Anomaly Detection")
    if len(numeric_cols) > 0:
        anomaly_model = IsolationForest(contamination=0.05, random_state=42)
        df["Anomaly"] = anomaly_model.fit_predict(df[numeric_cols])
        df["Anomaly"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})
        st.dataframe(df[df["Anomaly"] == "Anomaly"])  # Show anomalies
    
    # Select column for visualization
    selected_col = st.selectbox("Select a Column to Visualize", numeric_cols)
    
    if selected_col:
        # Histogram
        st.subheader(f"📊 Distribution of {selected_col}")
        fig = px.histogram(df, x=selected_col, nbins=20, title=f"Histogram of {selected_col}")
        st.plotly_chart(fig)
        
        # Line Chart
        st.subheader(f"📈 Trend of {selected_col}")
        fig = px.line(df, y=selected_col, title=f"Trend of {selected_col}")
        st.plotly_chart(fig)

st.sidebar.info("Upload a CSV file to analyze the data and detect anomalies!")