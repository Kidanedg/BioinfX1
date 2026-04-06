import streamlit as st
import pandas as pd
import numpy as np

st.title("📂 Force Field Dataset Explorer")

uploaded_file = st.file_uploader("Upload force field dataset")

if uploaded_file is not None:

    # =============================
    # 📖 READ FILE
    # =============================
    lines = uploaded_file.read().decode("utf-8").splitlines()

    st.subheader("📄 Raw Dataset")
    st.text_area("Dataset", "\n".join(lines), height=300)

    # =============================
    # 🧹 CLEAN DATA
    # =============================
    data = [
        line.split()
        for line in lines
        if not line.startswith("[") and line.strip()
    ]

    df = pd.DataFrame(data)

    st.subheader("📊 Preview")
    st.dataframe(df.head(20))

    # =============================
    # 🧠 AUTO COLUMN DETECTION
    # =============================
    st.markdown("## 🧠 Data Interpretation")

    try:
        df = df.apply(pd.to_numeric, errors='ignore')

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            st.success("✅ Numeric parameters detected")

            st.write("### 📈 Statistical Summary")
            st.dataframe(df[numeric_cols].describe())

            # =============================
            # 📊 VISUALIZATION
            # =============================
            st.write("### 📊 Parameter Distribution")

            col = st.selectbox("Select parameter", numeric_cols)

            st.line_chart(df[col])

        else:
            st.warning("⚠️ No numeric data detected")

    except Exception as e:
        st.error(f"Error processing data: {e}")

    # =============================
    # ⚛️ PHYSICS INTERPRETATION
    # =============================
    st.markdown("## ⚛️ Physical Meaning")

    st.markdown("""
    Typical force field datasets include:

    - **Bond parameters** → equilibrium distances and stiffness  
    - **Angle parameters** → preferred bond angles  
    - **Dihedral terms** → rotational barriers  
    - **Lennard-Jones parameters** → ε and σ  

    👉 These values directly feed into the total energy equation.
    """)

    # =============================
    # 🔬 SIMPLE ENERGY ESTIMATION
    # =============================
    st.markdown("## 🔬 Quick Energy Estimation")

    if len(numeric_cols) >= 2:

        col1, col2 = st.columns(2)

        with col1:
            r = st.slider("Distance (r)", 0.1, 5.0, 1.0)

        with col2:
            epsilon = st.slider("Epsilon (ε)", 0.01, 1.0, 0.1)

        sigma = 1.0  # default

        lj_energy = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

        st.metric("Lennard-Jones Energy", f"{lj_energy:.4f}")

        st.line_chart([
            4 * epsilon * ((sigma / x)**12 - (sigma / x)**6)
            for x in np.linspace(0.5, 3, 100)
        ])

    # =============================
    # 🧠 AI INSIGHT
    # =============================
    st.markdown("## 🧠 AI Insight")

    if len(numeric_cols) > 0:
        avg = df[numeric_cols].mean().mean()

        if avg > 1:
            st.success("🟢 Parameters suggest strong interactions")
        elif avg > 0.1:
            st.info("🟡 Moderate interaction strength")
        else:
            st.warning("🔴 Weak interaction parameters")

else:
    st.warning("Please upload a dataset file.")
