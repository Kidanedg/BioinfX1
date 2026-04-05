import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# =============================
# SAFE IMPORT
# =============================
try:
    from Bio.PDB import PDBParser
except:
    st.error("Install Biopython: pip install biopython")
    st.stop()

st.set_page_config(page_title="Biomolecular Teaching Platform", layout="wide")

# =============================
# LOGIN
# =============================
USERS = {"student": "1234", "admin": "admin"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(u) == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong login")
    st.stop()

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Theory", "Dataset", "Simulation", "Assignment", "Quiz", "Analytics", "Structure Analysis"]
)

# =============================
# FILE UPLOAD
# =============================
st.sidebar.markdown("### 📂 Upload Force Field")

atoms_file = st.sidebar.file_uploader("Atoms CSV", type="csv")
bonds_file = st.sidebar.file_uploader("Bonds CSV", type="csv")
angles_file = st.sidebar.file_uploader("Angles CSV", type="csv")
dihedrals_file = st.sidebar.file_uploader("Dihedrals CSV", type="csv")
nonbonded_file = st.sidebar.file_uploader("Nonbonded CSV", type="csv")

st.sidebar.markdown("### 🧭 Control Panel")

pdb_file = st.sidebar.file_uploader("Upload PDB", type="pdb")

# =============================
# 3D VIEWER (FIXED)
# =============================
def show_3d(pdb_string, style="stick"):

    styles = {
        "stick": '{"stick":{}}',
        "sphere": '{"sphere":{}}',
        "cartoon": '{"cartoon":{}}'
    }

    html = f"""
    <div id="viewer" style="width:100%; height:500px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
        viewer.addModel(`{pdb_string}`, "pdb");
        viewer.setStyle({styles[style]});
        viewer.zoomTo();
        viewer.render();
    </script>
    """

    components.html(html, height=500)

# =============================
# GEOMETRY
# =============================
def distance(a, b):
    return np.linalg.norm(a - b)

def detect_bonds(coords):
    return [(i, j) for i in range(len(coords))
            for j in range(i+1, len(coords))
            if distance(coords[i], coords[j]) < 1.8]

# =============================
# ENERGY
# =============================
def compute_energy(coords):
    E = 0
    bonds = detect_bonds(coords)

    for i, j in bonds:
        r = distance(coords[i], coords[j])
        E += 300 * (r - 1.5) ** 2

    return E

# =============================
# MAIN PAGES
# =============================

if page == "Theory":
    st.title("📘 Theory")
    st.write("Molecular mechanics, force fields, MD basics.")

elif page == "Dataset":
    st.title("📂 Dataset")
    st.write("Upload and explore molecular datasets.")

elif page == "Simulation":
    st.title("⚙ Simulation")

    if pdb_file:
        pdb_data = pdb_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mol", StringIO(pdb_data))
        atoms = list(structure.get_atoms())
        coords = np.array([a.get_coord() for a in atoms])

        st.success(f"{len(coords)} atoms loaded")

        style = st.selectbox("Style", ["stick", "sphere", "cartoon"])
        show_3d(pdb_data, style)

        if st.button("Compute Energy"):
            E = compute_energy(coords)
            st.metric("Energy", f"{E:.2f}")

    else:
        st.info("Upload a PDB file")

elif page == "Assignment":
    st.title("📝 Assignment")
    st.write("Student tasks and exercises.")

elif page == "Quiz":
    st.title("❓ Quiz")
    st.write("Interactive MCQs.")

elif page == "Analytics":
    st.title("📊 Analytics")
    st.write("Performance tracking.")

elif page == "Structure Analysis":
    st.title("🧬 Structure Analysis")
    st.write("Advanced structural metrics (RMSD, etc.)")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🔬 Biomolecular Teaching Platform | Clean Version")
