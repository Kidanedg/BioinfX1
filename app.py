import streamlit as st
import os
import numpy as np
import pandas as pd
import re
import py3Dmol

# Safe Biopython import
try:
    from Bio.PDB import PDBParser
except ImportError:
    st.error("Biopython not installed. Add it to requirements.txt")
    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Biomolecular Teaching Platform", layout="wide")

# -----------------------------
# LOGIN SYSTEM
# -----------------------------
USERS = {"student": "1234", "admin": "admin"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(username) == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
        else:
            st.error("Invalid credentials")
    st.stop()

# -----------------------------
# LOAD DATASETS
# -----------------------------
@st.cache_data
def load_data():
    try:
        atoms = pd.read_csv("data/atoms.csv")
        bonds = pd.read_csv("data/bonds.csv")
        angles = pd.read_csv("data/angles.csv")
        dihedrals = pd.read_csv("data/dihedrals.csv")
        nonbonded = pd.read_csv("data/nonbonded.csv")
        return atoms, bonds, angles, dihedrals, nonbonded
    except Exception as e:
        st.error(f"Dataset error: {e}")
        return None, None, None, None, None

atoms_df, bonds_df, angles_df, dihedrals_df, nonbonded_df = load_data()

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def safe_listdir(path):
    return os.listdir(path) if os.path.exists(path) else []

def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

def torsion(p1, p2, p3, p4):
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2/np.linalg.norm(b2))

    return np.degrees(np.arctan2(y, x))

# -----------------------------
# LIGAND MODEL
# -----------------------------
def ligand_properties(smiles):
    weights = {"C":12,"H":1,"O":16,"N":14,"S":32}
    tokens = re.findall(r'[A-Z][a-z]?', smiles)
    mw = sum(weights.get(t, 0) for t in tokens)
    logp = len(tokens) * 0.15
    return mw, logp

# -----------------------------
# FULL ENERGY FUNCTION
# -----------------------------
def compute_energy(coords, atom_types):

    bond_energy = 0
    angle_energy = 0
    dihedral_energy = 0
    vdw_energy = 0
    elec_energy = 0

    # Charges
    charge_map = dict(zip(atoms_df["atom_type"], atoms_df["charge"]))

    # LJ params
    lj_map = dict(zip(nonbonded_df["atom_type"],
                      zip(nonbonded_df["sigma"], nonbonded_df["epsilon"])))

    # ---------------- BONDS ----------------
    for i in range(len(coords)-1):
        r = distance(coords[i], coords[i+1])
        k = bonds_df["k"].iloc[0]
        r0 = bonds_df["r0"].iloc[0]
        bond_energy += k * (r - r0)**2

    # ---------------- ANGLES ----------------
    for i in range(len(coords)-2):
        theta = angle(coords[i], coords[i+1], coords[i+2])
        k = angles_df["k"].iloc[0]
        theta0 = angles_df["theta0"].iloc[0]
        angle_energy += k * (theta - theta0)**2

    # ---------------- DIHEDRALS ----------------
    for i in range(len(coords)-3):
        phi = torsion(coords[i], coords[i+1], coords[i+2], coords[i+3])
        Vn = dihedrals_df["Vn"].iloc[0]
        gamma = dihedrals_df["gamma"].iloc[0]
        n = dihedrals_df["n"].iloc[0]
        dihedral_energy += Vn * (1 + np.cos(np.radians(n*phi - gamma)))

    # ---------------- NONBONDED ----------------
    for i in range(len(coords)):
        for j in range(i+2, len(coords)):

            r = distance(coords[i], coords[j])

            if r < 8:
                at_i = atom_types[i]
                at_j = atom_types[j]

                sigma, epsilon = lj_map.get(at_i, (3.5, 0.2))

                vdw_energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

                qi = charge_map.get(at_i, 0)
                qj = charge_map.get(at_j, 0)

                elec_energy += (qi * qj) / r

    total = bond_energy + angle_energy + dihedral_energy + vdw_energy + elec_energy

    return {
        "Bond": bond_energy,
        "Angle": angle_energy,
        "Dihedral": dihedral_energy,
        "vdW": vdw_energy,
        "Electrostatic": elec_energy,
        "Total": total
    }

# -----------------------------
# DOCKING
# -----------------------------
def docking_score(protein_coords, ligand_coords):
    score = 0
    for p in protein_coords:
        for l in ligand_coords:
            r = distance(p, l)
            if r < 8:
                score += -1/r
    return score

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🧬 Biomolecular Teaching Platform")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Structure Analysis", "Force Field Explorer"])

# =====================================================
# STRUCTURE ANALYSIS
# =====================================================
if page == "Structure Analysis":

    st.sidebar.header("Structure Config")
    data_type = st.sidebar.selectbox("Dataset", ["DNA", "Protein"])
    data_path = "data/dna" if data_type == "DNA" else "data/proteins"

    files = safe_listdir(data_path)

    if not files:
        st.error("No PDB files found")
        st.stop()

    selected_file = st.sidebar.selectbox("Select File", files)

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("s", os.path.join(data_path, selected_file))
    except:
        st.error("Error loading PDB")
        st.stop()

    atoms = list(structure.get_atoms())
    coords = np.array([a.get_coord() for a in atoms])

    st.subheader("Structure Info")
    st.write(f"Atoms: {len(coords)}")

    # Ligand
    st.subheader("Ligand")
    smiles = st.text_input("SMILES", "CCO")
    mw, logp = ligand_properties(smiles)
    st.write(f"MW: {mw:.2f} | LogP: {logp:.2f}")

    # Assign atom types (simple)
    atom_types = ["C"] * len(coords)

    # ENERGY
    st.subheader("Energy")
    energy = compute_energy(coords, atom_types)

    st.metric("Total Energy", f"{energy['Total']:.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Bond", f"{energy['Bond']:.2f}")
    c2.metric("Angle", f"{energy['Angle']:.2f}")
    c3.metric("Dihedral", f"{energy['Dihedral']:.2f}")

    c4, c5 = st.columns(2)
    c4.metric("vdW", f"{energy['vdW']:.2f}")
    c5.metric("Electrostatic", f"{energy['Electrostatic']:.2f}")

    # 3D VIEW
    st.subheader("3D Viewer")
    try:
        with open(os.path.join(data_path, selected_file)) as f:
            pdb_data = f.read()

        view = py3Dmol.view(width=700, height=500)
        view.addModel(pdb_data, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()

        st.components.v1.html(view._make_html(), height=500)
    except:
        st.error("Visualization failed")

# =====================================================
# FORCE FIELD EXPLORER
# =====================================================
elif page == "Force Field Explorer":

    st.subheader("Force Field Data")

    tab = st.selectbox("Select", [
        "Atoms", "Bonds", "Angles", "Dihedrals", "Nonbonded"
    ])

    if tab == "Atoms":
        st.dataframe(atoms_df)
        st.bar_chart(atoms_df[["mass", "charge"]])

    elif tab == "Bonds":
        st.dataframe(bonds_df)
        st.bar_chart(bonds_df[["k"]])

    elif tab == "Angles":
        st.dataframe(angles_df)

    elif tab == "Dihedrals":
        st.dataframe(dihedrals_df)

    elif tab == "Nonbonded":
        st.dataframe(nonbonded_df)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🔬 Full AMBER Energy + Docking + Visualization")
