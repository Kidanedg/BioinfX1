import streamlit as st
import os
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Descriptors
import py3Dmol

st.set_page_config(page_title="Biomolecular Pipeline", layout="wide")

st.title("🧬 Advanced Biomolecular Interaction Pipeline")

# -----------------------------
# Utility Functions
# -----------------------------
def safe_listdir(path):
    return os.listdir(path) if os.path.exists(path) else []

def distance(a, b):
    return np.linalg.norm(a - b)

# -----------------------------
# Dataset Selection
# -----------------------------
st.sidebar.header("⚙️ Configuration")

data_type = st.sidebar.selectbox("Select Dataset Type", ["DNA", "Protein"])

if data_type == "DNA":
    data_path = "data/dna"
else:
    data_path = "data/proteins"

files = safe_listdir(data_path)

if not files:
    st.error(f"No files found in {data_path}")
    st.stop()

selected_file = st.sidebar.selectbox("Select Structure", files)

# -----------------------------
# Load Structure
# -----------------------------
parser = PDBParser(QUIET=True)

try:
    structure = parser.get_structure("structure", os.path.join(data_path, selected_file))
except:
    st.error("Error loading PDB file")
    st.stop()

atoms = list(structure.get_atoms())
coords = np.array([atom.get_coord() for atom in atoms])

st.subheader("📊 Structure Info")
st.write(f"Selected File: {selected_file}")
st.write(f"Total Atoms: {len(atoms)}")

# -----------------------------
# Ligand Dataset (.smi)
# -----------------------------
st.subheader("💊 Ligand Selection")

ligand_path = "data/ligands/ligands.smi"
ligands = []

if os.path.exists(ligand_path):
    with open(ligand_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles = parts[0]
                name = parts[1]
                ligands.append((name, smiles))

if ligands:
    ligand_names = [l[0] for l in ligands]
    selected_name = st.selectbox("Select Ligand", ligand_names)
    smiles = dict(ligands)[selected_name]
else:
    selected_name = "Custom"
    smiles = st.text_input("Enter Ligand SMILES", "CCO")

st.write(f"Selected Ligand: {selected_name}")
st.write(f"SMILES: {smiles}")

# -----------------------------
# Ligand Properties
# -----------------------------
mol = Chem.MolFromSmiles(smiles)

if mol:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    st.subheader("📈 Ligand Properties")
    st.write(f"Molecular Weight: {mw:.2f}")
    st.write(f"LogP: {logp:.2f}")
else:
    st.error("Invalid SMILES")
    st.stop()

# -----------------------------
# Interaction Calculations
# -----------------------------
st.subheader("⚙️ Interaction Calculations")

# Speed optimization (important for mobile)
max_atoms = 500
if len(coords) > max_atoms:
    idx = np.random.choice(len(coords), max_atoms, replace=False)
    coords = coords[idx]
    st.warning("Large structure detected → using sampled atoms for speed")

vdw_energy = 0.0
elec_energy = 0.0
hbond_count = 0

epsilon = 0.2
sigma = 3.5

for i in range(len(coords)):
    for j in range(i + 1, len(coords)):
        r = distance(coords[i], coords[j])

        if r < 8:
            # van der Waals (Lennard-Jones)
            vdw_energy += 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

            # Electrostatic (simplified)
            elec_energy += (-1) / r

            # Hydrogen bond (approx)
            if 2.5 < r < 3.5:
                hbond_count += 1

# -----------------------------
# Results
# -----------------------------
st.subheader("📊 Interaction Summary")

col1, col2, col3 = st.columns(3)

col1.metric("vdW Energy", f"{vdw_energy:.2f}")
col2.metric("Electrostatic Energy", f"{elec_energy:.2f}")
col3.metric("H-Bonds", hbond_count)

# -----------------------------
# Visualization
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
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🔬 Prototype for Biomolecular Interaction Analysis | Streamlit App")
