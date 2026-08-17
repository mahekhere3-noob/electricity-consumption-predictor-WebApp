"""
Electricity Consumption Predictor — Streamlit Web App
------------------------------------------------
A dashboard UI for the XGBoost electricity consumption model.

BEFORE YOU RUN THIS:
  1. Install dependencies:
       pip install streamlit pandas scikit-learn xgboost matplotlib
  2. Make sure "electricity_data.csv" is in this same folder.
  3. Run with:
       streamlit run app.py
     (NOT "python app.py" — Streamlit apps must be launched with the
     "streamlit run" command.)
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Electricity Consumption Predictor", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F3F6F9; }
    section[data-testid="stSidebar"] { background-color: #E1E8EF; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D3DCE6;
        border-radius: 10px;
        padding: 16px;
    }
    div.stButton > button {
        background-color: #F2A93B;
        color: #14213D;
        font-weight: 700;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #F5BC61;
        color: #14213D;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Train the model (cached so it only trains once per session)
# ---------------------------------------------------------------------------

FEATURES = ["Hour", "Day_of_Week", "Temperature", "Humidity", "Is_Weekend"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@st.cache_resource
def load_model():
    df = pd.read_csv("electricity_data.csv")

    X = df[FEATURES]
    y = df["Consumption_kWh"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    return model, r2, mae, len(df)


model, r2, mae, n_rows = load_model()


# ---------------------------------------------------------------------------
# Sidebar — Model Control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ Model Control")
    st.markdown("### Model Info")
    st.markdown("🔹 **Algorithm:** XGBoost Regressor")
    st.markdown(f"🔹 **Dataset:** Electricity Readings ({n_rows} rows)")
    st.markdown(f"🔹 **Test R² Score:** {r2:.3f}")
    st.markdown(f"🔹 **Test MAE:** {mae:.2f} kWh")

    st.markdown("---")
    st.markdown("### Settings")
    show_curve = st.checkbox("Show 24-hour load curve", value=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Electricity Consumption Predictor")
st.caption("Enter the time and weather conditions to estimate hourly electricity consumption.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Time")
    hour = st.slider("Hour of Day", 0, 23, 18)
    day_name = st.selectbox("Day of Week", DAY_NAMES, index=4)
    day_of_week = DAY_NAMES.index(day_name)
    is_weekend = 1 if day_of_week >= 5 else 0
    st.caption(f"Weekend: {'Yes' if is_weekend else 'No'} (set automatically from the day)")

with col2:
    st.markdown("### Weather")
    temperature = st.slider("Temperature (°C)", -5.0, 45.0, 27.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)

st.markdown("---")

if st.button("⚡ Predict Consumption", type="primary"):
    new_data = pd.DataFrame(
        [[hour, day_of_week, temperature, humidity, is_weekend]],
        columns=FEATURES,
    )
    pred = float(model.predict(new_data)[0])

    st.markdown(
        f"""
        <div style="background-color:#F2A93B22; border:2px solid #F2A93B;
                    border-radius:12px; padding:26px; text-align:center; margin-top:10px;">
            <div style="font-size:13px; letter-spacing:2px; color:#4B5A6B; text-transform:uppercase;">
                Predicted Consumption
            </div>
            <div style="font-size:58px; font-weight:bold; color:#B87A12; line-height:1.2;">
                {pred:.2f} <span style="font-size:26px;">kWh</span>
            </div>
            <div style="font-size:15px; color:#4B5A6B;">
                {day_name}, {hour:02d}:00 &middot; {temperature:.0f}\u00b0C &middot; {humidity:.0f}% humidity
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_curve:
        st.markdown("#### Predicted Load Curve — same day, every hour")
        hours = list(range(24))
        curve_data = pd.DataFrame(
            [[h, day_of_week, temperature, humidity, is_weekend] for h in hours],
            columns=FEATURES,
        )
        curve_pred = model.predict(curve_data)

        fig, ax = plt.subplots(figsize=(9, 3.4))
        fig.patch.set_facecolor("#F3F6F9")
        ax.set_facecolor("#F3F6F9")

        ax.plot(hours, curve_pred, color="#B87A12", linewidth=2.2, zorder=3)
        ax.fill_between(hours, curve_pred, color="#F2A93B", alpha=0.15, zorder=2)
        ax.scatter([hour], [pred], color="#B87A12", s=90, zorder=4, edgecolor="white", linewidth=1.5)

        ax.set_xlabel("Hour of day", fontsize=10, color="#4B5A6B")
        ax.set_ylabel("Predicted kWh", fontsize=10, color="#4B5A6B")
        ax.set_xticks(range(0, 24, 2))
        ax.tick_params(colors="#4B5A6B", labelsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#D3DCE6")
        ax.grid(axis="y", color="#D3DCE6", linewidth=0.7, zorder=0)
        fig.tight_layout()

        st.pyplot(fig)
        st.caption(
            f"Holding {day_name.lower()}, {temperature:.0f}\u00b0C and "
            f"{humidity:.0f}% humidity constant, this shows how predicted consumption "
            f"shifts across the day — the highlighted point is your selected hour."
        )
