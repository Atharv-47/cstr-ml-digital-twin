import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# rest of generate.py


# Constants
R = 8.314
N = 2000
F = 1.0  # m3/s

# Chemical database
CHEMICALS = {
    "Ethanol_Model": dict(k0=2e5, Ea=45000, dH=-75000, Cp=110, antoine=(8.20417, 1642.89, 230.3)),
    "Benzene_Model": dict(k0=8e4, Ea=52000, dH=-48000, Cp=135, antoine=(6.90565, 1211.033, 220.79))
}

# vectorized
def arrhenius(k0, Ea, T):
    return k0 * np.exp(-Ea / (R * T))

def psat_bar(A, B, C, T):
    T_C = T - 273.15
    return (10 ** (A - B / (T_C + C))) * 0.00133322

def conversion_cstr(k, tau):
    return (k * tau) / (1 + k * tau)

# Dataset generation
all_rows = []

for chem, p in CHEMICALS.items():

    # Sample operating conditions (VECTORIZED)
    T = np.random.uniform(300, 380, N)
    tau = np.random.uniform(30, 600, N)
    C0 = np.random.uniform(0.5, 2.0, N)
    P = np.random.uniform(1.0, 20.0, N)

    # Phase check (Antoine)
    Psat = psat_bar(*p["antoine"], T)
    mask = P > Psat  # liquid-only

    # Apply mask
    T, tau, C0, P = T[mask], tau[mask], C0[mask], P[mask]

    # Kinetics
    k = arrhenius(p["k0"], p["Ea"], T)
    X = conversion_cstr(k, tau)

    # Heat duty (negative => exothermic)
    Q = F * C0 * X * p["dH"]

    rows = np.column_stack([
        np.full_like(T, p["k0"]),
        np.full_like(T, p["Ea"]),
        np.full_like(T, p["dH"]),
        np.full_like(T, p["Cp"]),
        T, tau, C0, P,
        X, Q
    ])

    all_rows.append(rows)

# Save dataset
columns = ["k0", "Ea", "dH", "Cp","Temp_K", "Tau_s", "C0", "Pressure_bar","Conversion", "HeatDuty_W"]

df = pd.DataFrame(np.vstack(all_rows), columns=columns)
df.to_csv("data/master_chemical_data.csv", index=False)

print(" Dataset generated:", df.shape)
