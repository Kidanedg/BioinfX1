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
    ["Simulation", "Docking & Binding", "Structure Analysis"]
)

pdb_file = st.sidebar.file_uploader("Upload Protein PDB", type="pdb")
ligand_file = st.sidebar.file_uploader("Upload Ligand PDB", type="pdb")

# =============================
# VISUAL EXPLANATION (IMAGES)
# =============================
if page == "Structure Analysis":
    st.title("🧬 Protein Structure Insight")

    st.markdown("### 🧬 Protein Structural Levels")

    

    st.markdown("""
    - **Primary**: amino acid sequence  
    - **Secondary**: α-helix, β-sheet  
    - **Tertiary**: 3D folding  
    - **Quaternary**: multi-chain complexes  
    """)

# =============================
# 3D VIEWER (ADVANCED)
# =============================
def show_3d(pdb_string, style="cartoon"):

    html = f"""
    <div id="viewer" style="width:100%; height:500px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "black"}});
        viewer.addModel(`{pdb_string}`, "pdb");

        viewer.setStyle({{}}, {{
            cartoon: {{
                color: 'spectrum'
            }}
        }});

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

def neighbor_list(coords):
    return [(i, j) for i in range(len(coords))
            for j in range(i+1, len(coords))
            if distance(coords[i], coords[j]) < 6.0]

# =============================
# CHARGES
# =============================
def assign_charges(atoms):
    charge_map = {"O": -0.5, "N": -0.3, "C": 0.2, "H": 0.1}
    return np.array([charge_map.get(a.element.strip(), 0.0) for a in atoms])

# =============================
# HYDROGEN BONDS
# =============================
def detect_hbonds(coords, atoms):
    hbonds = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            if atoms[i].element == "O" and atoms[j].element == "H":
                if distance(coords[i], coords[j]) < 2.5:
                    hbonds.append((i, j))
    return hbonds

# =============================
# ENERGY
# =============================
def compute_energy(coords, atoms):

    bonds = detect_bonds(coords)
    pairs = neighbor_list(coords)
    charges = assign_charges(atoms)

    E_bond = E_vdw = E_coulomb = 0

    for i, j in bonds:
        r = distance(coords[i], coords[j])
        E_bond += 300 * (r - 1.5)**2

    for i, j in pairs:
        r = distance(coords[i], coords[j]) + 1e-6

        E_vdw += 4 * 0.2 * ((3.5/r)**12 - (3.5/r)**6)
        E_coulomb += (charges[i] * charges[j]) / r

    return bonds, pairs, {
        "Bond": E_bond,
        "VDW": E_vdw,
        "Coulomb": E_coulomb,
        "Total": E_bond + E_vdw + E_coulomb
    }

# =============================
# DOCKING (SIMPLE)
# =============================
def simple_docking(protein_coords, ligand_coords):
    center = protein_coords.mean(axis=0)
    ligand_center = ligand_coords.mean(axis=0)

    shift = center - ligand_center
    ligand_coords += shift

    return ligand_coords

# =============================
# MAIN: SIMULATION
# =============================
if page == "Simulation" and pdb_file:

    pdb_data = pdb_file.read().decode("utf-8")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", StringIO(pdb_data))
    atoms = list(structure.get_atoms())
    coords = np.array([a.get_coord() for a in atoms])

    st.success(f"{len(coords)} atoms loaded")

    show_3d(pdb_data)

    if st.button("Run Interaction Analysis"):

        bonds, pairs, energy = compute_energy(coords, atoms)
        hbonds = detect_hbonds(coords, atoms)

        st.metric("Total Energy", f"{energy['Total']:.2f}")

        fig, ax = plt.subplots()
        ax.bar(energy.keys(), energy.values())
        st.pyplot(fig)

        st.write(f"🔗 Bonds: {len(bonds)}")
        st.write(f"💧 Hydrogen Bonds: {len(hbonds)}")

# =============================
# DOCKING PAGE
# =============================
elif page == "Docking & Binding":

    st.title("⚛️ Docking Simulation")

    

    if pdb_file and ligand_file:

        prot_data = pdb_file.read().decode("utf-8")
        lig_data = ligand_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)

        prot = parser.get_structure("p", StringIO(prot_data))
        lig = parser.get_structure("l", StringIO(lig_data))

        p_atoms = list(prot.get_atoms())
        l_atoms = list(lig.get_atoms())

        p_coords = np.array([a.get_coord() for a in p_atoms])
        l_coords = np.array([a.get_coord() for a in l_atoms])

        docked = simple_docking(p_coords, l_coords)

        st.success("Docking completed (center-based)")

        st.write("📌 Ligand moved to protein center")

        st.metric("Estimated Binding Energy", f"{-np.linalg.norm(docked.mean(axis=0)):.2f}")

    else:
        st.info("Upload both protein and ligand")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🔬 Advanced Biomolecular Modeling & Docking Platform")
