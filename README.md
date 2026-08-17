# ⚡ Electricity Consumption Predictor Web App

A web app that predicts hourly electricity consumption (kWh) from time and weather conditions, using a trained XGBoost regression model.

Built as part of a hands-on machine learning mini-project series.

---

## 🔍 Overview

Pick an hour, a day of week, and current weather (temperature, humidity), and the app predicts electricity consumption for that moment — then plots a full 24-hour load curve so you can see how consumption is expected to shift across the day.

**[ Screenshot pending — insert a screenshot of the running app here ]**

---

## ✨ Features

- Real-time consumption prediction from time + weather inputs
- Weekend flag is derived automatically from the selected day — no contradictory inputs possible
- 24-hour predicted load curve with the selected hour highlighted, holding day and weather constant
- Sidebar with live model metadata — algorithm, dataset size, R² score, and Mean Absolute Error
- Clean two-panel dashboard UI, no ML background required to use it

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Model | XGBoost (`XGBRegressor`) |
| Data handling | pandas |
| Evaluation | scikit-learn |
| Charting | matplotlib |
| Web app | Streamlit |

---

## 📊 Dataset

- 400 hourly electricity readings
- Features: `Hour`, `Day_of_Week`, `Temperature`, `Humidity`, `Is_Weekend`
- Target: `Consumption_kWh`

---

## 🤖 Model

```python
XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    random_state=42
)
```

Trained with an 80/20 train-test split. Performance metrics (R², MAE) are computed live on startup and shown in the app's sidebar.

> **Note:** the dataset covers a limited range of temperature and humidity conditions. Predictions for extreme weather well outside that range are extrapolations and should be treated with caution.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/electricity-consumption-predictor.git
cd electricity-consumption-predictor
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn xgboost matplotlib
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
electricity-consumption-predictor/
├── app.py                                # Streamlit web app
├── electricity_consumption_xgboost.py    # Original standalone training script
├── electricity_data.csv                  # Dataset
└── .streamlit/
    └── config.toml                       # App theme
```

---

## ⚠️ Limitations

- Trained on a relatively small dataset (400 rows) — not validated against real utility or smart-meter data
- Predictions are illustrative and **not intended for actual grid planning, load balancing, or billing**
- Accuracy degrades for weather conditions far outside the training data's range

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
