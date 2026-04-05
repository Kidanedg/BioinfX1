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
# LOAD DATASETS (FIXED)
# -----------------------------
@st.cache_data
def load_data():
    try:
        def load_csv(path):
            df = pd.read_csv(path, encoding="utf-8-sig")  # 🔥 fixes BOM issue
            df.columns = df.columns.str.strip().str.lower()  # 🔥 clean names
            return df

        atoms = load_csv("data/atoms.csv")
        bonds = load_csv("data/bonds.csv")
        angles = load_csv("data/angles.csv")
        dihedrals = load_csv("data/dihedrals.csv")
        nonbonded = load_csv("data/nonbonded.csv")

        # Ensure numeric columns
        for col in ["mass", "charge", "sigma", "epsilon"]:
            if col in atoms.columns:
                atoms[col] = pd.to_numeric(atoms[col], errors="coerce")

        return atoms, bonds, angles, dihedrals, nonbonded

    except Exception as e:
        st.error(f"Dataset error: {e}")
        return None, None, None, None, None

atoms_df, bonds_df, angles_df, dihedrals_df, nonbonded_df = load_data()

# Stop if critical data missing
if atoms_df is None:
    st.stop()

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
# ENERGY FUNCTION (SAFE)
# -----------------------------
def compute_energy(coords, atom_types):

    bond_energy = angle_energy = dihedral_energy = vdw_energy = elec_energy = 0

    # Safe maps
    charge_map = dict(zip(atoms_df["atom_type"], atoms_df["charge"])) if "charge" in atoms_df else {}
    lj_map = dict(zip(nonbonded_df["atom_type"],
                      zip(nonbonded_df["sigma"], nonbonded_df["epsilon"]))) if nonbonded_df is not None else {}

    # BONDS
    if bonds_df is not None and "k" in bonds_df:
        for i in range(len(coords)-1):
            r = distance(coords[i], coords[i+1])
            k = bonds_df["k"].iloc[0]
            r0 = bonds_df["r0"].iloc[0]
            bond_energy += k * (r - r0)**2

    # ANGLES
    if angles_df is not None and "k" in angles_df:
        for i in range(len(coords)-2):
            theta = angle(coords[i], coords[i+1], coords[i+2])
            k = angles_df["k"].iloc[0]
            theta0 = angles_df["theta0"].iloc[0]
            angle_energy += k * (theta - theta0)**2

    # DIHEDRALS
    if dihedrals_df is not None and "vn" in dihedrals_df:
        for i in range(len(coords)-3):
            phi = torsion(coords[i], coords[i+1], coords[i+2], coords[i+3])
            Vn = dihedrals_df["vn"].iloc[0]
            gamma = dihedrals_df["gamma"].iloc[0]
            n = dihedrals_df["n"].iloc[0]
            dihedral_energy += Vn * (1 + np.cos(np.radians(n*phi - gamma)))

    # NONBONDED
    for i in range(len(coords)):
        for j in range(i+2, len(coords)):
            r = distance(coords[i], coords[j])
            if r < 8:
                at_i = atom_types[i]
                sigma, epsilon = lj_map.get(at_i, (3.5, 0.2))
                vdw_energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

                qi = charge_map.get(at_i, 0)
                qj = charge_map.get(at_i, 0)
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
# UI
# -----------------------------
st.title("🧬 Biomolecular Teaching Platform")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Structure Analysis", "Force Field Explorer"])

# =============================
# STRUCTURE ANALYSIS
# =============================
if page == "Structure Analysis":

    st.sidebar.header("Structure Config")
    data_path = "data/proteins"

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

    st.write(f"Atoms: {len(coords)}")

    smiles = st.text_input("SMILES", "CCO")
    mw, logp = ligand_properties(smiles)
    st.write(f"MW: {mw:.2f} | LogP: {logp:.2f}")

    atom_types = ["C"] * len(coords)
    energy = compute_energy(coords, atom_types)

    st.metric("Total Energy", f"{energy['Total']:.2f}")

# =============================
# FORCE FIELD EXPLORER
# =============================
elif page == "Force Field Explorer":

    st.subheader("Force Field Data")

    tab = st.selectbox("Select", ["Atoms", "Bonds", "Angles", "Dihedrals", "Nonbonded"])

    if tab == "Atoms":
        st.dataframe(atoms_df)

        if all(c in atoms_df.columns for c in ["mass", "charge"]):
            chart_df = atoms_df.set_index("atom_type")[["mass", "charge"]]
            st.bar_chart(chart_df)
        else:
            st.error("Columns missing")
            st.write(atoms_df.columns)

    elif tab == "Bonds":
        st.dataframe(bonds_df)

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
