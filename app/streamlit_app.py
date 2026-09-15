"""Streamlit demo application."""
import os
import numpy as np
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="DBCI-TNBC", layout="wide")

st.title("🧬 Deep Bayesian Causal Inference — TNBC pCR Prediction")
st.write("Enter multi-modal patient data (or use random demo data).")

col1, col2 = st.columns(2)

with col1:
    n_patients = st.number_input("Number of patients", 1, 32, 1)
    imaging_size = st.number_input("Imaging size (H=W)", 64, 512, 224, step=16)
    microbial_dim = st.number_input("Microbial features", 10, 5000, 500)
    transcriptomic_dim = st.number_input("Transcriptomic features", 10, 5000, 1000)

with col2:
    seed = st.number_input("Random seed", 0, 9999, 42)
    use_random = st.checkbox("Use random demo data", value=True)

if st.button("Run Prediction"):
    rng = np.random.default_rng(int(seed))
    imaging = rng.standard_normal(
        (int(n_patients), 1, int(imaging_size), int(imaging_size))
    ).astype("float32")
    microbial = rng.standard_normal((int(n_patients), int(microbial_dim))).astype("float32")
    transcriptomic = rng.standard_normal((int(n_patients), int(transcriptomic_dim))).astype("float32")

    payload = {
        "imaging": imaging.tolist(),
        "microbial": microbial.tolist(),
        "transcriptomic": transcriptomic.tolist(),
        "return_causal": True,
    }

    try:
        with st.spinner("Querying model..."):
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
            r.raise_for_status()
            result = r.json()

        st.success("Prediction complete")
        st.subheader("Results")
        for i, (p, u) in enumerate(zip(result["probabilities"], result["uncertainties"])):
            st.write(f"Patient {i+1}: pCR probability = **{p:.3f}** ± {u:.3f}")

        if result.get("causal_effects"):
            st.subheader("Causal Effects")
            for k, v in result["causal_effects"].items():
                st.write(f"{k}: {v}")

    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")