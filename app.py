import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# ----------------------------
# Load California Housing Dataset
# ----------------------------
@st.cache_data
def load_data():
    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["price"] = dataset.target
    return df

# ----------------------------
# Train Model
# ----------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics
    results = {
        "Train R2": round(r2_score(y_train, train_preds), 2),
        "Train MAE": round(mean_absolute_error(y_train, train_preds), 2),
        "Test R2": round(r2_score(y_test, test_preds), 2),
        "Test MAE": round(mean_absolute_error(y_test, test_preds), 2),
    }

    # Save trained model
    joblib.dump(model, "house_price_model.pkl")

    return model, results, X_train, y_train, train_preds, X_test, y_test, test_preds

# ----------------------------
# Prediction Function
# ----------------------------
def predict_price(model, features):
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)[0]
    return prediction

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🏠 California House Price Prediction", layout="wide")
st.title("🏠 California House Price Prediction with XGBoost")

menu = st.sidebar.radio("Navigation", ["Dataset", "Model Training", "Predict Price"])

df = load_data()

# ----------------------------
# Dataset Preview
# ----------------------------
if menu == "Dataset":
    st.subheader("📊 Dataset Preview")
    st.write(df.head())
    st.write("### 🔎 Correlation Heatmap")

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="Blues", ax=ax)
    st.pyplot(fig)

# ----------------------------
# Model Training & Evaluation
# ----------------------------
elif menu == "Model Training":
    st.subheader("⚡ Train & Evaluate Model")
    model, results, X_train, y_train, train_preds, X_test, y_test, test_preds = train_model(df)

    st.write("### 📈 Model Performance")
    st.json(results)

    st.write("### 🟢 Train: Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_train, train_preds, alpha=0.5)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    st.pyplot(fig)

    st.write("### 🔵 Test: Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_preds, alpha=0.5)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    st.pyplot(fig)

# ----------------------------
# Predict Price (User Input)
# ----------------------------
elif menu == "Predict Price":
    st.subheader("🔮 Predict House Price")

    try:
        model = joblib.load("house_price_model.pkl")
    except:
        st.error("⚠️ Model not trained yet! Go to **Model Training** first.")
        st.stop()

    st.write("Enter house details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        MedInc = st.number_input("Median Income (MedInc)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        HouseAge = st.number_input("House Age", min_value=1, max_value=100, value=20, step=1)
        AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=5.0, step=0.1)

    with col2:
        AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
        Population = st.number_input("Population", min_value=1, max_value=10000, value=500, step=10)
        AveOccup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)

    with col3:
        Latitude = st.number_input("Latitude", min_value=30.0, max_value=50.0, value=34.0, step=0.01)
        Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-110.0, value=-118.0, step=0.01)

    if st.button("Predict Price"):
        features = {
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude,
        }
        price = predict_price(model, features)
        st.success(f"🏡 Predicted House Price: **${price*100000:.2f}**")