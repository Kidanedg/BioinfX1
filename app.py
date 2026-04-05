import streamlit as st
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Descriptors
import py3Dmol

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

# -----------------------------
# FORCE FIELD PARSER
# -----------------------------
def load_forcefield(file="forcefield_dataset.txt"):
    sections = {}
    current = None

    if not os.path.exists(file):
        return {}

    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("["):
                current = line.strip("[]")
                sections[current] = []
            else:
                sections[current].append(line.split())

    return sections

# -----------------------------
# SAVE SCORES
# -----------------------------
def save_score(user, score):
    file = "scores.csv"
    new = pd.DataFrame([[user, score]], columns=["User", "Score"])

    if os.path.exists(file):
        old = pd.read_csv(file)
        df = pd.concat([old, new])
    else:
        df = new

    df.to_csv(file, index=False)

# -----------------------------
# DOCKING FUNCTION
# -----------------------------
def docking_score(protein_coords, ligand_coords):
    vdw = 0
    elec = 0
    hbond = 0
    hydrophobic = 0

    epsilon = 0.2
    sigma = 3.5

    for p in protein_coords:
        for l in ligand_coords:
            r = distance(p, l)

            if r < 8:
                vdw += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
                elec += -1 / r

                if 2.5 < r < 3.5:
                    hbond += -1

                if r < 5:
                    hydrophobic += -0.1

    total = vdw + elec + hbond + hydrophobic

    return vdw, elec, hbond, hydrophobic, total

# -----------------------------
# TITLE
# -----------------------------
st.title("🧬 Biomolecular Teaching Platform (AMBER + Docking + LMS)")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Configuration")

# Show dataset
if os.path.exists("forcefield_dataset.txt"):
    with open("forcefield_dataset.txt") as f:
        st.sidebar.text_area("Force Field Dataset", f.read(), height=300)

# -----------------------------
# STRUCTURE SELECTION
# -----------------------------
data_type = st.sidebar.selectbox("Dataset Type", ["DNA", "Protein"])
data_path = "data/dna" if data_type == "DNA" else "data/proteins"

files = safe_listdir(data_path)

if not files:
    st.error("No PDB files found")
    st.stop()

selected_file = st.sidebar.selectbox("Select Structure", files)

# -----------------------------
# LOAD STRUCTURE
# -----------------------------
parser = PDBParser(QUIET=True)

try:
    structure = parser.get_structure("structure", os.path.join(data_path, selected_file))
except:
    st.error("Error loading structure")
    st.stop()

atoms = list(structure.get_atoms())
coords = np.array([atom.get_coord() for atom in atoms])

st.subheader("📊 Structure Info")
st.write(f"File: {selected_file}")
st.write(f"Atoms: {len(atoms)}")

# -----------------------------
# LIGAND INPUT
# -----------------------------
st.subheader("💊 Ligand")

smiles = st.text_input("Enter SMILES", "CCO")
mol = Chem.MolFromSmiles(smiles)

if mol:
    st.write(f"MW: {Descriptors.MolWt(mol):.2f}")
    st.write(f"LogP: {Descriptors.MolLogP(mol):.2f}")
else:
    st.error("Invalid SMILES")
    st.stop()

# -----------------------------
# STUDENT PREDICTION
# -----------------------------
st.subheader("🎓 Assignment")

student_guess = st.number_input("Predict Total Energy", value=0.0)

# -----------------------------
# ENERGY ENGINE
# -----------------------------
st.subheader("⚙️ Energy Calculation")

bond_energy = 0
angle_energy = 0
vdw_energy = 0
elec_energy = 0

kb = 300
r0 = 1.5
k_theta = 40
theta0 = 109.5
epsilon = 0.2
sigma = 3.5

# Limit size
if len(coords) > 300:
    coords = coords[np.random.choice(len(coords), 300, replace=False)]

# Bonds
for i in range(len(coords)-1):
    r = distance(coords[i], coords[i+1])
    bond_energy += kb * (r - r0)**2

# Angles
for i in range(len(coords)-2):
    th = angle(coords[i], coords[i+1], coords[i+2])
    angle_energy += k_theta * (th - theta0)**2

# Nonbonded
for i in range(len(coords)):
    for j in range(i+2, len(coords)):
        r = distance(coords[i], coords[j])
        if r < 8:
            vdw_energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
            elec_energy += (-1) / r

total_energy = bond_energy + angle_energy + vdw_energy + elec_energy

# -----------------------------
# RESULTS
# -----------------------------
st.subheader("📊 Energy Results")

col1, col2, col3 = st.columns(3)
col1.metric("Bond", f"{bond_energy:.2f}")
col2.metric("Angle", f"{angle_energy:.2f}")
col3.metric("vdW", f"{vdw_energy:.2f}")

col4, col5 = st.columns(2)
col4.metric("Electrostatic", f"{elec_energy:.2f}")
col5.metric("Total Energy", f"{total_energy:.2f}")

# -----------------------------
# DOCKING
# -----------------------------
st.subheader("🧪 Docking Simulation")

if st.button("Run Docking"):
    ligand_coords = coords[:50]
    vdw, elec, hbond, hydro, dock_total = docking_score(coords, ligand_coords)

    st.write("### Docking Breakdown")
    st.write(f"vdW: {vdw:.2f}")
    st.write(f"Electrostatic: {elec:.2f}")
    st.write(f"H-bond: {hbond}")
    st.write(f"Hydrophobic: {hydro:.2f}")
    st.success(f"Docking Score: {dock_total:.2f}")

# -----------------------------
# STUDENT FEEDBACK
# -----------------------------
st.subheader("🎯 Feedback")

error = abs(student_guess - total_energy)
score = max(0, 100 - error)

st.write(f"Your Prediction: {student_guess}")
st.write(f"Actual Energy: {total_energy:.2f}")
st.write(f"Error: {error:.2f}")
st.write(f"Score: {score:.2f}")

if st.button("Submit Score"):
    save_score(st.session_state.user, score)
    st.success("Score saved!")

# -----------------------------
# EXPORT
# -----------------------------
if os.path.exists("scores.csv"):
    with open("scores.csv", "rb") as f:
        st.download_button("Download Scores", f, "scores.csv")

# -----------------------------
# 3D VISUALIZATION
# -----------------------------
st.subheader("🧬 3D Structure")

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

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🔬 Full Teaching Platform: Force Fields + Docking + LMS")
