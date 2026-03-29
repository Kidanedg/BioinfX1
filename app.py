
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
import py3Dmol

st.title("🧬 Simple Biomolecular Interaction Pipeline")

# Upload protein
protein_file = st.file_uploader("Upload Protein (PDB)", type=["pdb"])

# Input ligand
ligand_smiles = st.text_input("Enter Ligand SMILES", "CCO")

# Step 1: Process ligand
if ligand_smiles:
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        st.subheader("Ligand Properties")
        st.write(f"Molecular Weight: {mw:.2f}")
        st.write(f"LogP: {logp:.2f}")

        # Fake docking score (simple formula)
        score = -0.1 * mw + logp
        st.subheader("Docking Score (Simulated)")
        st.write(score)

# Step 2: Visualize protein
if protein_file:
    pdb_data = protein_file.read().decode("utf-8")

    st.subheader("Protein Visualization")

    view = py3Dmol.view(width=500, height=400)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()

    st.components.v1.html(view._make_html(), height=400)
