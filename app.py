import streamlit as st
import os
import numpy as np
import pandas as pd
import re
import py3Dmol
import matplotlib.pyplot as plt

# =============================
# SAFE IMPORT
# =============================
try:
    from Bio.PDB import PDBParser
except:
    st.error("Install Biopython")
    st.stop()

st.set_page_config(page_title="Biomolecular Platform", layout="wide")

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
        else:
            st.error("Wrong login")
    st.stop()

# =============================
# LOAD FORCE FIELD
# =============================
@st.cache_data
def load_ff():
    try:
        def load_csv(p):
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip().str.lower()
            return df

        return (
            load_csv("data/atoms.csv"),
            load_csv("data/bonds.csv"),
            load_csv("data/angles.csv"),
            load_csv("data/dihedrals.csv"),
            load_csv("data/nonbonded.csv"),
        )
    except:
        return None, None, None, None, None

atoms_df, bonds_df, angles_df, dihedrals_df, nonbonded_df = load_ff()

# =============================
# UPLOAD FF
# =============================
st.sidebar.markdown("### 📂 Upload Force Field")

def load_uploaded(file):
    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        return df
    return None

atoms_df = load_uploaded(st.sidebar.file_uploader("Atoms CSV")) or atoms_df
bonds_df = load_uploaded(st.sidebar.file_uploader("Bonds CSV")) or bonds_df
angles_df = load_uploaded(st.sidebar.file_uploader("Angles CSV")) or angles_df
dihedrals_df = load_uploaded(st.sidebar.file_uploader("Dihedrals CSV")) or dihedrals_df
nonbonded_df = load_uploaded(st.sidebar.file_uploader("Nonbonded CSV")) or nonbonded_df

# =============================
# GEOMETRY
# =============================
def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ba, bc = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) /
           (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1)))

def torsion(p1, p2, p3, p4):
    b1, b2, b3 = p2-p1, p3-p2, p4-p3
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    return np.degrees(np.arctan2(
        np.dot(np.cross(n1, n2), b2/np.linalg.norm(b2)),
        np.dot(n1, n2)
    ))

# =============================
# AUTO BONDS
# =============================
def detect_bonds(coords, cutoff=1.8):
    bonds = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if distance(coords[i], coords[j]) < cutoff:
                bonds.append((i, j))
    return bonds

# =============================
# NEIGHBOR LIST
# =============================
def build_neighbor_list(coords, cutoff=8.0):
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if distance(coords[i], coords[j]) < cutoff:
                pairs.append((i, j))
    return pairs

# =============================
# CHARGES
# =============================
def assign_charges(atoms):
    charges = []
    for atom in atoms:
        element = atom.element.strip()
        charge_map = {"O": -0.5, "N": -0.3, "C": 0.2, "H": 0.1, "S": -0.2}
        charges.append(charge_map.get(element, 0.0))
    return np.array(charges)

# =============================
# ENERGY
# =============================
def compute_energy(coords, atoms=None):

    E_bond = E_angle = E_dihedral = E_vdw = E_coulomb = 0.0

    # Bonds
    if bonds_df is not None:
        for _, row in bonds_df.iterrows():
            i, j = int(row['i']), int(row['j'])
            k = row.get('k', 300)
            r0 = row.get('r0', 1.5)
            r = distance(coords[i], coords[j])
            E_bond += k * (r - r0) ** 2
    else:
        for i, j in detect_bonds(coords):
            r = distance(coords[i], coords[j])
            E_bond += 300 * (r - 1.5) ** 2

    # Angles
    if angles_df is not None:
        for _, row in angles_df.iterrows():
            i, j, k_idx = int(row['i']), int(row['j']), int(row['k'])
            theta = angle(coords[i], coords[j], coords[k_idx])
            E_angle += 40 * (theta - 109.5) ** 2

    # Dihedrals
    if dihedrals_df is not None:
        for _, row in dihedrals_df.iterrows():
            i, j, k_idx, l = map(int, [row['i'], row['j'], row['k'], row['l']])
            phi = torsion(coords[i], coords[j], coords[k_idx], coords[l])
            E_dihedral += 1 * (1 + np.cos(np.radians(3 * phi)))

    # Nonbonded
    pairs = build_neighbor_list(coords)
    charges = assign_charges(atoms) if atoms else np.zeros(len(coords))

    for i, j in pairs:
        r = distance(coords[i], coords[j]) + 1e-6
        vdw = 4 * 0.2 * ((3.5/r)**12 - (3.5/r)**6)
        coulomb = (charges[i] * charges[j]) / r
        E_vdw += vdw
        E_coulomb += coulomb

    total = E_bond + E_angle + E_dihedral + E_vdw + E_coulomb

    return {
        "Bond": E_bond,
        "Angle": E_angle,
        "Dihedral": E_dihedral,
        "Van der Waals": E_vdw,
        "Coulomb": E_coulomb,
        "Total": total
    }

# =============================
# MD ENGINE
# =============================
def compute_forces(coords):
    forces = np.zeros_like(coords)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            r_vec = coords[i] - coords[j]
            r = np.linalg.norm(r_vec) + 1e-6
            f = (1 / r**2) * (r_vec / r)
            forces[i] += f
            forces[j] -= f
    return forces

def velocity_verlet(coords, velocities, dt=0.001):
    f = compute_forces(coords)
    coords += velocities * dt + 0.5 * f * dt**2
    f_new = compute_forces(coords)
    velocities += 0.5 * (f + f_new) * dt
    return coords, velocities

# =============================
# PARAMETER LEARNING
# =============================
def fit_bond_k(coords):
    k = 300.0
    bonds = detect_bonds(coords)

    for _ in range(100):
        grad = sum(2 * (distance(coords[i], coords[j]) - 1.5)**2 for i, j in bonds)
        k -= 1e-5 * grad

    return k

# =============================
# 3D VIEW
# =============================
def show_3d(pdb_string):
    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_string, "pdb")
    view.setStyle({"stick": {}})
    view.zoomTo()
    return view.show()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🧭 Control Panel")

dataset = st.sidebar.selectbox("Dataset", ["PDB", "DNA", "Force Field", "Ligand"])
mode = st.sidebar.selectbox("Mode", ["Structure", "Energy", "Simulation", "Explorer"])

# =============================
# MAIN
# =============================
st.title("🧬 Biomolecular Teaching Platform")

if dataset in ["PDB", "DNA"]:
    folder = "data/proteins" if dataset == "PDB" else "data/dna"
    files = os.listdir(folder) if os.path.exists(folder) else []

    if files:
        file = st.selectbox("Select file", files)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", os.path.join(folder, file))

        atoms = list(structure.get_atoms())
        coords = np.array([a.get_coord() for a in atoms])

        if mode == "Structure":
            with open(os.path.join(folder, file)) as f:
                show_3d(f.read())

        elif mode == "Energy":
            energy = compute_energy(coords, atoms)
            st.metric("Total Energy", f"{energy['Total']:.2f}")

            fig, ax = plt.subplots()
            ax.bar(list(energy.keys())[:-1], list(energy.values())[:-1])
            st.pyplot(fig)

        elif mode == "Simulation":
            if "vel" not in st.session_state:
                st.session_state.vel = np.zeros_like(coords)

            if st.button("Run MD Step"):
                coords, st.session_state.vel = velocity_verlet(coords, st.session_state.vel)
                st.success("Step done")

        elif mode == "Explorer":
            if st.button("Learn Bond Parameter"):
                k = fit_bond_k(coords)
                st.success(f"Learned k: {k:.2f}")

elif dataset == "Force Field":
    st.dataframe(atoms_df)

elif dataset == "Ligand":
    smiles = st.text_input("SMILES", "CCO")
    tokens = re.findall(r'[A-Z][a-z]?', smiles)
    st.write("Atoms:", tokens)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🔬 Full Molecular Mechanics Platform")
