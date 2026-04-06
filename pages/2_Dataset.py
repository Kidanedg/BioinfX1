import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.title("📂 Force Field Dataset Explorer")

uploaded_file = st.file_uploader("Upload force field dataset (CSV or PDB)")

if uploaded_file is not None:

    # =============================
    # 📖 READ FILE
    # =============================
    file_text = uploaded_file.read().decode("utf-8")
    lines = file_text.splitlines()

    st.subheader("📄 Raw Dataset")
    st.text_area("Dataset", file_text, height=250)

    # =============================
    # 🧠 DETECT FILE TYPE
    # =============================
    is_pdb = any(line.startswith(("ATOM", "HETATM")) for line in lines)
    is_csv = "," in lines[0]

    # =========================================================
    # 🧬 PDB MODE (STRUCTURE ANALYSIS)
    # =========================================================
    if is_pdb:
        st.success("🧬 Detected PDB structure")

        atoms = []
        coords = []

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                atoms.append(parts[2])
                coords.append([float(parts[6]), float(parts[7]), float(parts[8])])

        coords = np.array(coords)

        df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
        df["Atom"] = atoms

        st.subheader("📊 Atomic Coordinates")
        st.dataframe(df)

        # =============================
        # 📏 DISTANCE ANALYSIS
        # =============================
        st.markdown("## 📏 Distance Analysis")

        if len(coords) > 1:
            dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

            st.metric("Average Distance", f"{np.mean(dists):.3f}")
            st.metric("Max Distance", f"{np.max(dists):.3f}")

            st.line_chart(dists.flatten())

        # =============================
        # 🔬 LJ ENERGY
        # =============================
        st.markdown("## 🔬 Lennard-Jones Energy")

        epsilon = st.slider("Epsilon (ε)", 0.01, 1.0, 0.1)
        sigma = st.slider("Sigma (σ)", 0.5, 3.0, 1.0)

        if len(coords) > 1:
            dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) + 1e-6
            lj_energy = np.sum(4 * epsilon * ((sigma / dists)**12 - (sigma / dists)**6))

            st.metric("Total LJ Energy", f"{lj_energy:.4f}")

    # =========================================================
    # 📊 CSV MODE (FORCE FIELD ANALYSIS)
    # =========================================================
    elif is_csv:
        st.success("📊 Detected CSV force field dataset")

        df = pd.read_csv(StringIO(file_text))

        st.subheader("📊 Clean Dataset")
        st.dataframe(df)

        # =============================
        # 🧠 DATA INTERPRETATION
        # =============================
        numeric_cols = df.select_dtypes(include=np.number).columns

        st.markdown("## 🧠 Data Interpretation")

        if len(numeric_cols) > 0:
            st.success("✅ Numeric parameters detected")

            st.write("### 📈 Statistical Summary")
            st.dataframe(df[numeric_cols].describe())

            # =============================
            # 📊 VISUALIZATION
            # =============================
            col = st.selectbox("Select parameter", numeric_cols)
            st.line_chart(df[col])

        else:
            st.warning("⚠️ No numeric data detected")

        # =============================
        # ⚛️ INTERACTION ENERGY
        # =============================
        st.markdown("## ⚛️ Atom Interaction Energy")

        atom1 = st.selectbox("Atom 1", df["atom"])
        atom2 = st.selectbox("Atom 2", df["atom"], index=1)

        r = st.slider("Distance (r)", 0.5, 6.0, 2.0)

        a1 = df[df["atom"] == atom1].iloc[0]
        a2 = df[df["atom"] == atom2].iloc[0]

        # =============================
        # 🧪 MIXING RULES
        # =============================
        sigma = (a1["sigma"] + a2["sigma"]) / 2
        epsilon = np.sqrt(a1["epsilon"] * a2["epsilon"])

        # =============================
        # ⚡ COULOMB ENERGY
        # =============================
        k_e = 138.935456  # kJ·mol⁻¹·nm·e⁻²
        coulomb = k_e * (a1["charge"] * a2["charge"]) / r

        # =============================
        # 🌌 LENNARD-JONES
        # =============================
        lj = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

        total_energy = lj + coulomb

        # =============================
        # 📊 OUTPUT
        # =============================
        col1, col2, col3 = st.columns(3)

        col1.metric("LJ Energy", f"{lj:.4f}")
        col2.metric("Coulomb Energy", f"{coulomb:.4f}")
        col3.metric("Total Energy", f"{total_energy:.4f}")

        st.info(f"""
        σ = {sigma:.3f}  
        ε = {epsilon:.3f}  
        """)

        # =============================
        # 📈 ENERGY CURVE
        # =============================
        st.markdown("## 📈 Energy vs Distance")

        r_values = np.linspace(0.5, 6, 100)
        lj_curve = 4 * epsilon * ((sigma / r_values)**12 - (sigma / r_values)**6)
        coul_curve = k_e * (a1["charge"] * a2["charge"]) / r_values

        energy_df = pd.DataFrame({
            "Distance": r_values,
            "LJ": lj_curve,
            "Coulomb": coul_curve,
            "Total": lj_curve + coul_curve
        })

        st.line_chart(energy_df.set_index("Distance"))

    # =========================================================
    # ❌ UNKNOWN FORMAT
    # =========================================================
    else:
        st.error("❌ Unsupported file format")

    # =============================
    # 🧠 AI INSIGHT
    # =============================
    st.markdown("## 🧠 AI Insight")

    st.info("""
    ✔ CSV → Parameter-based force field (ε, σ, charge)  
    ✔ PDB → Structure-based coordinates  

    👉 Together they define full molecular simulations.
    """)

else:
    st.warning("Please upload a dataset file.")
