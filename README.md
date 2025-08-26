# 🏠 California House Price Prediction App

This project is a **Streamlit web app** that predicts **California house prices** using the **XGBoost Regressor** model.  
It uses the **California Housing dataset** from `sklearn.datasets`.

---

## 🚀 Features

- ✅ Explore the dataset with preview & heatmap  
- ✅ Train & evaluate an **XGBoost model**  
- ✅ View **R² score** and **MAE** for train & test sets  
- ✅ Visualize **Actual vs Predicted prices**  
- ✅ Input your own house features & get an instant prediction  

---

## 📊 Live Demo

🔗 Deployed on Render:  
👉 [California House Price Prediction App](https://house-price-prediction-app-m0z3.onrender.com/)

---

## ⚡ Installation & Run Locally

Clone the repo:

```bash
git clone https://github.com/Tom-1508/house-price-app.git
cd house-price-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

---

## 📂 Project Structure

```
house-price-app/
│── streamlit_app.py       # Main Streamlit app
│── house_price_model.pkl  # Saved trained model
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (UI)
* **XGBoost** (Regression Model)
* **Scikit-learn** (Dataset & metrics)
* **Matplotlib & Seaborn** (Visualizations)

---

## 🌍 Deployment (Render)

1. Push this repo to GitHub  
2. Go to [Render](https://render.com) → **New Web Service**  
3. Connect your GitHub repo  
4. Set up with:  
   - **Build Command:** `pip install -r requirements.txt`  
   - **Start Command:**  
     ```bash
     streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
     ```  
5. Click **Deploy** 🚀  

---

## ✨ Example Prediction

Input features (example):

* MedInc: `6.0`
* HouseAge: `30`
* AveRooms: `5.5`
* AveBedrms: `1.2`
* Population: `800`
* AveOccup: `3.2`
* Latitude: `34.5`
* Longitude: `-118.2`

**Output:**
🏡 Predicted House Price: **\$250,000 – \$300,000 (approx.)**

---

🙌 Made with ❤️ using **Streamlit & XGBoost**
