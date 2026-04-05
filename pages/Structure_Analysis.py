import streamlit as st
import numpy as np
import pandas as pd

# Optional 3D viewer
try:
    import py3Dmol
except:
    py3Dmol = None

st.title("🧬 Structure Analysis (Advanced)")

# =========================================================
# 🔹 PDB PARSER
# =========================================================
def parse_pdb(lines):
    coords = []
    elements = []
    atom_names = []

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip()
                name = line[12:16].strip()

                coords.append([x, y, z])
                elements.append(elem if elem else "C")
                atom_names.append(name)
            except:
                continue

    return np.array(coords), elements, atom_names

# =========================================================
# 🔹 DISTANCE MATRIX
# =========================================================
def compute_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(coords[i] - coords[j])

    return D

# =========================================================
# 🔹 BOND DETECTION (simple cutoff)
# =========================================================
def detect_bonds(coords, threshold=1.8):
    bonds = []

    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < threshold:
                bonds.append((i, j, d))

    return bonds

# =========================================================
# 🔹 ANGLE COMPUTATION
# =========================================================
def compute_angles(coords, bonds):
    angles = []

    for i, j, _ in bonds:
        for k, l, _ in bonds:
            if j == k and i != l:
                v1 = coords[i] - coords[j]
                v2 = coords[l] - coords[j]

                denom = np.linalg.norm(v1)*np.linalg.norm(v2)
                if denom == 0:
                    continue

                theta = np.degrees(
                    np.arccos(np.clip(np.dot(v1, v2)/denom, -1, 1))
                )

                angles.append((i, j, l, theta))

    return angles

# =========================================================
# 🔹 FILE INPUT
# =========================================================
st.markdown("### 📂 Upload Structure")

uploaded_file = st.file_uploader("Upload PDB file", type=["pdb"])

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    coords, elements, atom_names = parse_pdb(lines)

    st.session_state.coords = coords
    st.session_state.elements = elements

    st.success(f"✅ Loaded {len(coords)} atoms")

# =========================================================
# 🔹 DEMO STRUCTURE
# =========================================================
st.markdown("### 🧪 Demo Molecule")

if st.button("Load Demo (Helix-like Structure)"):
    t = np.linspace(0, 4*np.pi, 20)
    coords = np.column_stack((np.cos(t), np.sin(t), t/2))
    elements = ["C"] * len(coords)

    st.session_state.coords = coords
    st.session_state.elements = elements
    st.success("✅ Demo helix loaded")
    st.rerun()

# =========================================================
# 🔹 LOAD FROM SESSION
# =========================================================
coords = st.session_state.get("coords", None)
elements = st.session_state.get("elements", None)

if coords is None:
    st.warning("⚠️ No structure loaded")
    st.stop()

# =========================================================
# 📊 BASIC SUMMARY
# =========================================================
st.markdown("### 📊 Structure Summary")

df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
df["Element"] = elements

st.dataframe(df.head(25))
st.info(f"Total atoms: {len(coords)}")

# Element distribution
st.markdown("### 🧪 Element Distribution")
st.bar_chart(pd.Series(elements).value_counts())

# =========================================================
# 📏 DISTANCE MATRIX
# =========================================================
st.markdown("### 📏 Distance Matrix")

D = compute_distance_matrix(coords)

if st.checkbox("Show Distance Matrix"):
    st.dataframe(pd.DataFrame(D).round(2))

# =========================================================
# 🔗 BONDS
# =========================================================
st.markdown("### 🔗 Bond Detection")

bond_threshold = st.slider("Bond Threshold (Å)", 1.0, 3.0, 1.8)

bonds = detect_bonds(coords, bond_threshold)

st.write(f"Detected Bonds: {len(bonds)}")

if st.checkbox("Show Bonds"):
    bond_df = pd.DataFrame(bonds, columns=["Atom1", "Atom2", "Distance"])
    st.dataframe(bond_df.head(20))

# =========================================================
# 📐 ANGLES
# =========================================================
st.markdown("### 📐 Bond Angles")

angles = compute_angles(coords, bonds)

st.write(f"Computed Angles: {len(angles)}")

if st.checkbox("Show Angles"):
    angle_df = pd.DataFrame(angles, columns=["i", "j", "k", "Angle"])
    st.dataframe(angle_df.head(20))

# =========================================================
# 🎥 3D VISUALIZATION
# =========================================================
st.markdown("### 🎥 3D Viewer")

if py3Dmol:

    style = st.selectbox("Style", ["stick", "sphere", "line"])

    pdb_str = ""
    for i, (x, y, z) in enumerate(coords):
        elem = elements[i]
        pdb_str += f"ATOM  {i:5d} {elem:>2s} MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"

    viewer = py3Dmol.view(width=700, height=500)
    viewer.addModel(pdb_str, "pdb")

    if style == "stick":
        viewer.setStyle({"stick": {}})
    elif style == "sphere":
        viewer.setStyle({"sphere": {}})
    else:
        viewer.setStyle({"line": {}})

    viewer.zoomTo()

    st.components.v1.html(viewer._make_html(), height=500)

# =========================================================
# 🚀 EXPORT TO SIMULATION
# =========================================================
st.markdown("---")
st.success("🚀 Structure ready for Simulation Engine")
