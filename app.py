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
# SIDEBAR
# =============================
st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Simulation", "Structure Analysis"]
)

pdb_file = st.sidebar.file_uploader("Upload PDB", type="pdb")

# =============================
# 3D VIEWER
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

def detect_bonds(coords, cutoff=1.8):
    return [(i, j) for i in range(len(coords))
            for j in range(i+1, len(coords))
            if distance(coords[i], coords[j]) < cutoff]

def neighbor_list(coords, cutoff=8.0):
    return [(i, j) for i in range(len(coords))
            for j in range(i+1, len(coords))
            if distance(coords[i], coords[j]) < cutoff]

# =============================
# CHARGES
# =============================
def assign_charges(atoms):
    charge_map = {"O": -0.5, "N": -0.3, "C": 0.2, "H": 0.1, "S": -0.2}
    return np.array([charge_map.get(a.element.strip(), 0.0) for a in atoms])

# =============================
# ENERGY PIPELINE
# =============================
def compute_energy(coords, atoms):

    bonds = detect_bonds(coords)
    pairs = neighbor_list(coords)
    charges = assign_charges(atoms)

    E_bond = 0
    E_vdw = 0
    E_coulomb = 0

    # Bond energy
    for i, j in bonds:
        r = distance(coords[i], coords[j])
        E_bond += 300 * (r - 1.5)**2

    # Nonbonded
    for i, j in pairs:
        r = distance(coords[i], coords[j]) + 1e-6

        # Lennard-Jones
        sigma = 3.5
        epsilon = 0.2
        E_vdw += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

        # Coulomb
        E_coulomb += (charges[i] * charges[j]) / r

    total = E_bond + E_vdw + E_coulomb

    return bonds, pairs, {
        "Bond": E_bond,
        "Van der Waals": E_vdw,
        "Coulomb": E_coulomb,
        "Total": total
    }

# =============================
# MAIN
# =============================
if pdb_file:

    pdb_data = pdb_file.read().decode("utf-8")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", StringIO(pdb_data))
    atoms = list(structure.get_atoms())
    coords = np.array([a.get_coord() for a in atoms])

    st.success(f"{len(coords)} atoms loaded")

    style = st.selectbox("Visualization Style", ["stick", "sphere", "cartoon"])
    show_3d(pdb_data, style)

    # =============================
    # PIPELINE RUN
    # =============================
    if st.button("🚀 Run Interaction Pipeline"):

        bonds, pairs, energy = compute_energy(coords, atoms)

        # ENERGY OUTPUT
        st.subheader("⚡ Energy Results")
        st.metric("Total Energy (kcal/mol)", f"{energy['Total']:.2f}")

        # ENERGY PLOT
        fig, ax = plt.subplots()
        ax.bar(energy.keys(), energy.values())
        ax.set_title("Energy Components")
        st.pyplot(fig)

        # =============================
        # BOND TABLE
        # =============================
        st.subheader("🔗 Bonds Detected")
        bond_df = pd.DataFrame(bonds, columns=["Atom1", "Atom2"])
        st.dataframe(bond_df)

        # =============================
        # INTERACTION TABLE
        # =============================
        st.subheader("🌐 Nonbonded Interactions")
        inter_df = pd.DataFrame(pairs[:200], columns=["Atom1", "Atom2"])
        st.dataframe(inter_df)

else:
    st.info("👈 Upload a PDB file")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🔬 Biomolecular Interaction & Energy Platform")
