# Industrial CSTR Digital Twin (ML-Based)

An interactive digital twin for a Continuous Stirred Tank Reactor (CSTR) using physics-based modeling and machine learning. 
It predicts **conversion (%)** and **heat duty (kW)** for selected chemicals under user-defined operating conditions.


## Reactions Modeled
| Chemical | Reaction|
|----------|---------------|
| Ethanol  | Catalytic Dehydration: C2H5OH → C2H4 + H2O |
| Benzene  | Nitration: C6H6 + HNO3 → C6H5NO2 + H2O |

**Chemical constants used:**

| Chemical | k0 (1/s) | Ea (J/mol) | ΔH (J/mol) | Cp (J/mol·K) |
|----------|----------|------------|------------|---------------|
| Ethanol  | 2.0×10⁵ | 45,000     | -75,000    | 110           |
| Benzene  | 8.0×10⁴ | 52,000     | -48,000    | 135           |


##  Features

- Steady-state CSTR modeling  
- Arrhenius first-order kinetics  
- Predicts **conversion** and **heat duty**  
- Supports multiple operating parameters: Temperature (K), Pressure (bar), Residence time (τ, s), Reactant/Solvent ratio  
- Safety warnings for extreme conditions  
- PDF report generation with constants and assumptions  
- Sensitivity plots and Levenspiel analysis
    

##  Machine Learning

- Algorithm: **Random Forest Regressor** (n_estimators=200, max_depth=12)  
- Regression-based surrogate model for conversion & heat duty  
- Inputs (features): Chemical ID, Temperature, Pressure, Residence time, Feed ratio, Chemical constants (k0, Ea, ΔH, Cp)  
- Outputs (targets): Conversion (%), Heat duty (kW)  
- R² ≈0.997 (Note: This is due to the dataset being **synthetically generated from physics-based formulas**.
  It reflects that the ML model closely approximates the simulator outputs.
- Model trained on synthetic physics-based dataset  
- Scalable to new chemicals (via dataset regeneration)


##  Assumptions

- Steady-state operation  
- Ideal mixing (CSTR)  
- Liquid-phase reaction  
- First-order irreversible kinetics  
- Constant physical properties  


##  How to Run Locally

1. Install dependencies: `pip install -r requirements.txt`  
2. Generate dataset: `python src/generate_master.py`  
3. Train ML model: `python src/train_model.py`  
4. Launch app: `streamlit run src/app.py`  


##  Notes

- Generated datasets and trained models are **excluded from GitHub**  
- Directories (`data/` and `models/`) are created automatically at runtime  
- Ensure **Python ≥3.8** and required packages are installed
  

##  Disclaimer

This project is for **academic and demonstration purposes only**. Not intended for plant design or safety-critical use.
