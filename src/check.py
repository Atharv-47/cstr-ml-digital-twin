import numpy as np
import joblib

model = joblib.load("data/reactor_model.pkl")
scaler = joblib.load("data/scaler.pkl")
x1 = scaler.transform([[2e5, 45000, -75000, 110, 350, 120, 1.5, 10]])
x2 = scaler.transform([[8e4, 52000, -48000, 135, 350, 120, 1.5, 10]])

print("Ethanol-like:", model.predict(x1))
print("Benzene-like:", model.predict(x2))
