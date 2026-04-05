import streamlit as st
import numpy as np
import pandas as pd

# =========================================================
# 🔥 SESSION STATE INIT (CRITICAL)
# =========================================================
if "coords" not in st.session_state:
    st.session_state.coords = None

if "elements" not in st.session_state:
    st.session_state.elements = None

# Optional 3D viewer
try:
    import py3Dmol
except:
    py3Dmol = None

st.title("🧬 Structure Analysis (Advanced)")

# =========================================================
# 🔹 PDB PARSER (ROBUST)
# =========================================================
def parse_pdb(lines):
    coords, elements, atom_names = [], [], []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                elem = line[76:78].strip()
                if not elem:
                    elem = line[12:14].strip()

                coords.append([x, y, z])
                elements.append(elem if elem else "C")
                atom_names.append(line[12:16].strip())
            except:
                continue

    return np.array(coords), elements, atom_names


# =========================================================
# 🔹 CENTER + NORMALIZE
# =========================================================
def center_structure(coords):
    center = np.mean(coords, axis=0)
    return coords - center


# =========================================================
# 🔹 DISTANCE MATRIX
# =========================================================
def compute_distance_matrix(coords):
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)


# =========================================================
# 🔹 BOND DETECTION
# =========================================================
def detect_bonds(coords, threshold=1.8):
    bonds = []
    n = len(coords)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < threshold:
                bonds.append((i, j, d))

    return bonds


# =========================================================
# 🔹 ANGLES
# =========================================================
def compute_angles(coords, bonds):
    angles = []

    for i, j, _ in bonds:
        for k, l, _ in bonds:
            if j == k and i != l:
                v1 = coords[i] - coords[j]
                v2 = coords[l] - coords[j]

                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                if denom == 0:
                    continue

                theta = np.degrees(
                    np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))
                )

                angles.append((i, j, l, theta))

    return angles


# =========================================================
# 🔹 DIHEDRAL ANGLES (NEW 🔥)
# =========================================================
def compute_dihedrals(coords, bonds):
    dihedrals = []

    for i, j, _ in bonds:
        for k, l, _ in bonds:
            if j == k:
                for m, n, _ in bonds:
                    if l == m and len({i, j, l, n}) == 4:

                        p0, p1, p2, p3 = coords[i], coords[j], coords[l], coords[n]

                        b0 = -1.0 * (p1 - p0)
                        b1 = p2 - p1
                        b2 = p3 - p2

                        b1 /= np.linalg.norm(b1)

                        v = b0 - np.dot(b0, b1) * b1
                        w = b2 - np.dot(b2, b1) * b1

                        x = np.dot(v, w)
                        y = np.dot(np.cross(b1, v), w)

                        angle = np.degrees(np.arctan2(y, x))

                        dihedrals.append((i, j, l, n, angle))

    return dihedrals


# =========================================================
# 🔹 FILE INPUT
# =========================================================
st.markdown("### 📂 Upload Structure")

uploaded_file = st.file_uploader("Upload PDB file", type=["pdb"])

if uploaded_file:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    coords, elements, atom_names = parse_pdb(lines)

    coords = center_structure(coords)

    st.session_state.coords = coords
    st.session_state.elements = elements

    st.success(f"✅ Loaded {len(coords)} atoms")


# =========================================================
# 🔹 DEMO
# =========================================================
st.markdown("### 🧪 Demo Molecule")

if st.button("Load Demo (Helix)"):
    t = np.linspace(0, 4 * np.pi, 30)
    coords = np.column_stack((np.cos(t), np.sin(t), t / 2))
    coords = center_structure(coords)

    elements = ["C"] * len(coords)

    st.session_state.coords = coords
    st.session_state.elements = elements

    st.success("✅ Demo loaded")
    st.rerun()


# =========================================================
# 🔹 LOAD SESSION
# =========================================================
coords = st.session_state.coords
elements = st.session_state.elements

if coords is None:
    st.warning("⚠️ No structure loaded")
    st.stop()


# =========================================================
# 📊 SUMMARY
# =========================================================
st.markdown("### 📊 Structure Summary")

df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
df["Element"] = elements

st.dataframe(df.head(25))
st.info(f"Total atoms: {len(coords)}")

st.markdown("### 🧪 Element Distribution")
st.bar_chart(pd.Series(elements).value_counts())


# =========================================================
# 📏 DISTANCE
# =========================================================
st.markdown("### 📏 Distance Matrix")

D = compute_distance_matrix(coords)

if st.checkbox("Show Distance Matrix"):
    st.dataframe(pd.DataFrame(D).round(2))


# =========================================================
# 🔗 BONDS
# =========================================================
st.markdown("### 🔗 Bonds")

threshold = st.slider("Bond Threshold", 1.0, 3.0, 1.8)
bonds = detect_bonds(coords, threshold)

st.write(f"Detected Bonds: {len(bonds)}")

if st.checkbox("Show Bonds"):
    st.dataframe(pd.DataFrame(bonds, columns=["i", "j", "Distance"]))


# =========================================================
# 📐 ANGLES
# =========================================================
st.markdown("### 📐 Angles")

angles = compute_angles(coords, bonds)
st.write(f"Angles: {len(angles)}")

if st.checkbox("Show Angles"):
    st.dataframe(pd.DataFrame(angles, columns=["i", "j", "k", "Angle"]))


# =========================================================
# 🔄 DIHEDRALS
# =========================================================
st.markdown("### 🔄 Dihedrals (Torsion)")

dihedrals = compute_dihedrals(coords, bonds)
st.write(f"Dihedrals: {len(dihedrals)}")

if st.checkbox("Show Dihedrals"):
    st.dataframe(pd.DataFrame(dihedrals,
                 columns=["i", "j", "k", "l", "Angle"]))


# =========================================================
# 🎥 3D VIEWER
# =========================================================
st.markdown("### 🎥 3D Viewer")

if py3Dmol:
    style = st.selectbox("Style", ["stick", "sphere", "line"])

    pdb_str = ""
    for i, (x, y, z) in enumerate(coords):
        e = elements[i]
        pdb_str += f"ATOM  {i:5d} {e:>2s} MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {e:>2s}\n"

    viewer = py3Dmol.view(width=800, height=500)
    viewer.addModel(pdb_str, "pdb")

    viewer.setStyle({style: {}})
    viewer.zoomTo()

    st.components.v1.html(viewer._make_html(), height=500)
else:
    st.warning("py3Dmol not installed")


# =========================================================
# 🚀 EXPORT
# =========================================================
st.markdown("---")

if st.button("🚀 Send to Simulation Engine"):
    st.session_state.ready_for_sim = True
    st.success("Sent to Simulation Engine ✅")

st.success("Structure ready ✔")
