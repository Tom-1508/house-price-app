````markdown
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

## 📊 Demo

🔗 Live Demo (Hugging Face Spaces):  
👉 [California House Price App](https://huggingface.co/spaces/Tom-1508/california-house-price)

---

## ⚡ Installation & Run Locally

Clone the repo:

```bash
git clone https://github.com/Tom-1508/house-price-app.git
cd house-price-app
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
house-price-app/
│── app.py                 # Main Streamlit app
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

## 🌍 Deployment (Hugging Face Spaces)

1. Push this repo to GitHub
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
3. Create a new Space → Select **Streamlit** SDK
4. Connect your GitHub repo (or upload manually)
5. Hugging Face installs dependencies from `requirements.txt`
6. App goes live 🚀

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

```