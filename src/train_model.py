
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv("data/master_chemical_data.csv").dropna()

# Features & targets
X = df[["k0", "Ea", "dH", "Cp", "Temp_K", "Tau_s", "C0", "Pressure_bar"]]
Y = df[["Conversion", "HeatDuty_W"]]

# Split data BEFORE scaling
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Overall R² (average across both outputs)
print(f"Overall R² = {model.score(X_test, Y_test):.3f}")

# Evaluate each target separately
for i, target in enumerate(Y.columns):

    actual = Y_test[target].to_numpy()
    predicted = Y_pred[:, i]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\nEvaluation for {target}")
    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"R²   = {r2:.6f}")

# Save model and scaler
joblib.dump(model, "data/reactor_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")
