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
    # 🧠 DETECT FILE TYPE
    # =============================
    is_pdb = any(line.startswith("ATOM") or line.startswith("HETATM") for line in lines)

    # =============================
    # 🧬 PDB MODE (SMART HANDLING)
    # =============================
    if is_pdb:
        st.success("🧬 Detected PDB structure (not a force field file)")

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
        # 🔬 ENERGY ESTIMATION (REAL)
        # =============================
        st.markdown("## 🔬 Lennard-Jones Energy")

        epsilon = st.slider("Epsilon (ε)", 0.01, 1.0, 0.1)
        sigma = st.slider("Sigma (σ)", 0.5, 3.0, 1.0)

        if len(coords) > 1:
            dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2) + 1e-6

            lj_energy = np.sum(4 * epsilon * ((sigma / dists)**12 - (sigma / dists)**6))

            st.metric("Total LJ Energy", f"{lj_energy:.4f}")

    # =============================
    # 📂 GENERIC DATASET MODE
    # =============================
    else:

        data = [
            line.split()
            for line in lines
            if not line.startswith("[") and line.strip()
        ]

        df = pd.DataFrame(data)

        st.subheader("📊 Preview")
        st.dataframe(df.head(20))

        st.markdown("## 🧠 Data Interpretation")

        # ✅ FIX: safe numeric conversion
        df_numeric = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

        numeric_cols = df_numeric.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            st.success("✅ Numeric parameters detected")

            st.write("### 📈 Statistical Summary")
            st.dataframe(df_numeric[numeric_cols].describe())

            col = st.selectbox("Select parameter", numeric_cols)
            st.line_chart(df_numeric[col])

        else:
            st.warning("⚠️ No numeric data detected")

        # =============================
        # 🔬 ENERGY ESTIMATION
        # =============================
        st.markdown("## 🔬 Quick Energy Estimation")

        if len(numeric_cols) >= 2:

            r = st.slider("Distance (r)", 0.1, 5.0, 1.0)
            epsilon = st.slider("Epsilon (ε)", 0.01, 1.0, 0.1)

            sigma = 1.0

            lj_energy = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

            st.metric("LJ Energy", f"{lj_energy:.4f}")

    # =============================
    # 🧠 AI INSIGHT
    # =============================
    st.markdown("## 🧠 AI Insight")

    st.info("""
    ✔ PDB → Structure-based physics (distances, geometry)  
    ✔ Dataset → Parameter-based force field  

    👉 You uploaded a STRUCTURE, not a force field file.
    """)

else:
    st.warning("Please upload a dataset file.")
