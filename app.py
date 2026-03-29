import streamlit as st
import os
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import Descriptors
import py3Dmol

st.title("🧬 Advanced Biomolecular Interaction Pipeline")

# -----------------------------
# Load dataset
# -----------------------------
data_path = "data/proteins"
files = os.listdir(data_path)

selected_file = st.selectbox("Select Protein/DNA", files)

# Load structure
parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", os.path.join(data_path, selected_file))

atoms = list(structure.get_atoms())

st.write(f"Total atoms: {len(atoms)}")

# -----------------------------
# Ligand input
# -----------------------------
smiles = st.text_input("Enter Ligand SMILES", "CCO")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        st.subheader("Ligand Properties")
        st.write(f"Molecular Weight: {mw:.2f}")
        st.write(f"LogP: {logp:.2f}")

# -----------------------------
# Interaction calculations
# -----------------------------
def distance(a, b):
    return np.linalg.norm(a - b)

vdw_energy = 0
elec_energy = 0
hbond_count = 0

epsilon = 0.2
sigma = 3.5

coords = [atom.get_coord() for atom in atoms]

for i in range(len(coords)):
    for j in range(i+1, len(coords)):
        r = distance(coords[i], coords[j])

        if r < 8:  # cutoff
            # Lennard-Jones
            vdw = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
            vdw_energy += vdw

            # Coulomb (simplified)
            q1, q2 = 1, -1  # dummy charges
            elec = (q1*q2)/r
            elec_energy += elec

            # Hydrogen bond (approx rule)
            if 2.5 < r < 3.5:
                hbond_count += 1

# -----------------------------
# Display results
# -----------------------------
st.subheader("Interaction Summary")

st.write(f"van der Waals Energy: {vdw_energy:.2f}")
st.write(f"Electrostatic Energy: {elec_energy:.2f}")
st.write(f"Hydrogen Bonds (approx): {hbond_count}")

# -----------------------------
# Visualization
# -----------------------------
with open(os.path.join(data_path, selected_file)) as f:
    pdb_data = f.read()

view = py3Dmol.view(width=500, height=400)
view.addModel(pdb_data, "pdb")
view.setStyle({"cartoon": {"color": "spectrum"}})
view.zoomTo()

st.components.v1.html(view._make_html(), height=400)
