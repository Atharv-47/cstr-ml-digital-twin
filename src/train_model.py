import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("data/master_chemical_data.csv").dropna()

# Features & targets
X = df[["k0","Ea","dH","Cp","Temp_K","Tau_s","C0","Pressure_bar"]]
Y = df[["Conversion","HeatDuty_W"]]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, Y_train)

# Evaluate
print(f"R² = {model.score(X_test, Y_test):.3f}")
joblib.dump(model, "data/reactor_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")
