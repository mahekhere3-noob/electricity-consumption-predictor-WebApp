import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load Dataset
# Expected columns: Hour, Day_of_Week, Temperature, Humidity, Is_Weekend, Consumption_kWh
data = pd.read_csv("electricity_data.csv")

# Features and Target
X = data[["Hour", "Day_of_Week", "Temperature", "Humidity", "Is_Weekend"]]
y = data["Consumption_kWh"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.08,
    random_state=42
)
model.fit(X_train, y_train)

# Test Accuracy
prediction = model.predict(X_test)

print("Model R² Score:", r2_score(y_test, prediction))
print("Mean Absolute Error:", mean_absolute_error(y_test, prediction), "kWh")

# User Prediction
hour = float(input("Enter Hour of Day (0-23): "))
day_of_week = float(input("Enter Day of Week (0=Mon ... 6=Sun): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
is_weekend = float(input("Is it a weekend? (1 = Yes, 0 = No): "))

new_data = [[hour, day_of_week, temperature, humidity, is_weekend]]

result = model.predict(new_data)

print(f"\nPredicted Electricity Consumption: {result[0]:.2f} kWh")
