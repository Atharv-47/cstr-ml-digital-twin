import streamlit as st, pandas as pd, joblib, numpy as np, matplotlib.pyplot as plt
from fpdf import FPDF

# Load model & scaler
try:
    model = joblib.load("data/reactor_model.pkl")
    scaler = joblib.load("data/scaler.pkl")
except:
    st.error("🚨 Missing model files. Run train_model.py first.")
    st.stop()

st.set_page_config(page_title="Industrial CSTR Digital Twin", layout="wide")

st.markdown("""<style>
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {display:none;}
</style>""", unsafe_allow_html=True)

# Chemical properties
CHEM_PROPS = {
    "Ethanol":  (2e5, 45000, -75000, 110),
    "Benzene":  (8e4, 52000, -48000, 135)
}

tab_sim, tab_doc = st.tabs([" Reactor Simulator", " Technical Documentation"])

# SIMULATOR TAB
with tab_sim:
    st.title("REACTOR MODEL (CSTR)")
    st.markdown("---")

    col_img, col_input = st.columns([1.2, 1])

    with col_img:
        try:
            st.image("src/cstr.png", caption="Process Flow Diagram", use_container_width=True)
        except:
            st.warning("CSTR image missing")

    with col_input:
        chem = st.selectbox("Select Feed Reactant", CHEM_PROPS.keys())
        temp = st.number_input("Inlet Temperature (K)", 280.0, 500.0, 350.0)
        st.caption("(NOTE:Operational temperature is restricted to maintain liquid-phase conditions and ensure predictions remain within the validated domain of the reactor model. ""Operation outside this range may lead to unreliable results and is therefore not supported.)")

        pres = st.number_input("Reactor Pressure (bar)", 1.0, 25.0, 5.0)
        st.caption("(NOTE: Operating pressure is constrained to avoid phase transition and to ensure consistency with the conditions under which the reactor model was developed and validated.)")


    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        ratio = st.slider("Reactant Fraction in Feed", 0.1, 1.0, 0.5)
        C0_total_map = {"Ethanol": 1.0, "Benzene": 1.5}
        st.caption("NOTE: Fraction is converted internally to inlet concentration (C₀).")



    with c2:
        tau = st.slider("Residence Time τ (s)", 10, 1000, 300)
        st.latex(r"\tau = \frac{V}{v_0}")

    # Prediction Engine
    k0, Ea, dH, Cp = CHEM_PROPS[chem]
    C0 = ratio * C0_total_map[chem]
    C0_total = C0_total_map[chem]

    x = scaler.transform([[k0, Ea, dH, Cp, temp, tau, C0, pres]])
    conv, heat = model.predict(x)[0]

    st.markdown("---")

    # Safety Logic
    if heat < -5e5:
        st.error(f"🚨 CRITICAL: Excessive exothermic heat release ({heat/1000:.1f} kW)")
    elif conv < 0.10:
        st.warning("⚠️ Low conversion — increase temperature or residence time")
    else:
        st.success("✅ Reactor operating safely")

    # Results
    l, r = st.columns(2)
    with l:
        st.metric("Conversion (%)", f"{conv*100:.2f}")
        st.metric("Heat Duty (kW)", f"{heat/1000:.2f}")

    # Optimization
    with r:
        if st.button(" Find Temperature for 90% Conversion"):
            t_scan = np.linspace(280, 500, 400)
            scan = scaler.transform([[k0, Ea, dH, Cp, t, tau, C0, pres] for t in t_scan])
            X_scan = model.predict(scan)[:, 0]
            opt_T = t_scan[np.abs(X_scan - 0.9).argmin()]
            st.info(f"Optimal Temperature ≈ {opt_T:.1f} K")

        def generate_report():
            pdf = FPDF(); pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(0,10,f"CSTR PERFORMANCE REPORT ({chem})",ln=True,align="C")
            pdf.ln(5)
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0,8,
    f"Conversion: {conv*100:.2f}%\n"
    f"Heat Duty: {heat/1000:.2f} kW\n"
    f"Temperature: {temp} K\n"
    f"Pressure: {pres} bar\n"
    f"Residence Time: {tau} s\n\n"
    f"--- Chemical Constants Used ---\n"
    f"k0 (1/s): {k0:.2e}\n"
    f"Ea (J/mol): {Ea:.0f}\n"
    f"dH (J/mol): {dH:.0f}\n"
    f"C0_total (mol/L): {C0_total:.2f}\n"
    f"Cp (J/mol*K): {Cp:.0f}\n\n"

    f"--- Model Assumptions ---\n"
    f"- Steady-state CSTR\n"
    f"- Perfect mixing\n"
    f"- Liquid-phase reaction\n"
    f"- First-order irreversible kinetics\n"
    f"- Arrhenius temperature dependence\n"
    f"- Constant physical properties\n"
)

            return bytes(pdf.output())

        st.download_button("📥 Download PDF Report",
                           generate_report(),
                           file_name=f"{chem}_CSTR_Report.pdf")

    # -------------------------
    # Plots (FIXED INPUTS)
    # -------------------------
    st.subheader("📈 Kinetic & Design Analysis")
    T_plot = np.linspace(290, 480, 60)
    plot = scaler.transform([[k0, Ea, dH, Cp, t, tau, C0, pres] for t in T_plot])
    Xp = model.predict(plot)[:,0]
    rate = (C0 * Xp) / tau

    fig, ax = plt.subplots(1,3, figsize=(16,5))

    ax[0].plot(T_plot, Xp*100)
    ax[0].set_title("Conversion vs Temperature")
    ax[0].set_xlabel("Temperature (K)")
    ax[0].set_ylabel("Conversion (%)")

    ax[1].plot(T_plot, rate)
    ax[1].set_title("Reaction Rate")
    ax[1].set_xlabel("Temperature (K)")
    ax[1].set_ylabel("Rate (-rA)")

    m = rate > 1e-6
    ax[2].plot(Xp[m]*100, 1/rate[m])
    ax[2].set_title("Levenspiel Plot")
    ax[2].set_xlabel("Conversion (%)")
    ax[2].set_ylabel("1 / (-rA)")

    plt.tight_layout()
    fig.subplots_adjust(left=0.07, right=0.98, wspace=0.35)
    st.pyplot(fig)


# DOCUMENTATION TAB
with tab_doc:
    st.header("📘 Theoretical Framework & Assumptions")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Reaction Kinetics")
        if chem == "Ethanol":
            st.markdown("**Ethanol Dehydration:**")
            st.latex(r"C_2H_5OH \xrightarrow{H_2SO_4} C_2H_4 + H_2O")
        else:
            st.markdown("**Benzene Nitration:**")
            st.latex(r"C_6H_6 + HNO_3 \rightarrow C_6H_5NO_2 + H_2O")
        
        st.latex(r"k = A \cdot e^{-\frac{E_a}{RT}}")
        st.subheader("2. Mass Balance")
        st.latex(r"V = \frac{F_{A0} \cdot X}{-r_A}")
    with col_b:
        st.subheader("3. Core Assumptions")
        st.info("- Perfect Mixing\n- Steady State\n- Liquid Phase Operation\n- Modified Rackett Density Model")
    
    st.divider()
    