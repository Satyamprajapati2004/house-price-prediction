import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("model.pkl")

# Load dataset
df = pd.read_csv("train.csv")

st.set_page_config(page_title="House AI Dashboard", layout="wide")

# ---------------- TITLE ----------------
st.title("🏡 California House Price AI Dashboard")

st.markdown("### Machine Learning powered real estate prediction system")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Input Controls")

MedInc = st.sidebar.slider("Median Income", 0.0, 15.0, 3.0)
HouseAge = st.sidebar.slider("House Age", 1, 52, 20)
AveRooms = st.sidebar.slider("Avg Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Avg Bedrooms", 1.0, 5.0, 1.0)
Population = st.sidebar.slider("Population", 0, 5000, 1500)
AveOccup = st.sidebar.slider("Avg Occupancy", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -118.0)

# ---------------- PREDICTION ----------------
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

prediction = model.predict(input_data)[0]

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Median Income", MedInc)

with col2:
    st.metric("House Age", HouseAge)

with col3:
    st.metric("Predicted Price", f"{prediction:.2f}")

st.success(f"🏠 Estimated House Price: {prediction:.2f}")

st.markdown("---")

# ---------------- DATA INSIGHTS ----------------
st.subheader("📊 Data Insights")

numeric_df = df.select_dtypes(include=['number'])

col4, col5 = st.columns(2)

with col4:
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with col5:
    st.markdown("### Feature Distribution")

    feature = st.selectbox("Select Feature", numeric_df.columns)

    fig2, ax2 = plt.subplots()
    ax2.hist(numeric_df[feature], bins=25)
    st.pyplot(fig2)