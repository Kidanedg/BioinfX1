import streamlit as st
import os
import numpy as np
import pandas as pd
import re
import py3Dmol

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
            df = pd.read_csv(p, encoding="utf-8-sig")
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
# UTILITIES
# =============================
def distance(a,b): return np.linalg.norm(a-b)

def angle(a,b,c):
    ba, bc = a-b, c-b
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)), -1,1)))

def torsion(p1,p2,p3,p4):
    b1,b2,b3 = p2-p1,p3-p2,p4-p3
    n1,n2 = np.cross(b1,b2), np.cross(b2,b3)
    n1/=np.linalg.norm(n1); n2/=np.linalg.norm(n2)
    return np.degrees(np.arctan2(np.dot(np.cross(n1,n2), b2/np.linalg.norm(b2)), np.dot(n1,n2)))

# =============================
# ENERGY
# =============================
def compute_energy(coords):

    E = 0
    for i in range(len(coords)-1):
        r = distance(coords[i], coords[i+1])
        E += 300*(r-1.5)**2

    return E

# =============================
# 3D VIEWER
# =============================
def show_3d(pdb_string):
    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_string, "pdb")
    view.setStyle({"stick":{}})
    view.zoomTo()
    return view.show()

# =============================
# SIDEBAR NAVIGATION
# =============================
st.sidebar.title("🧭 Control Panel")

dataset = st.sidebar.selectbox(
    "Dataset Type",
    ["Force Field", "PDB", "DNA", "Ligand"]
)

mode = st.sidebar.selectbox(
    "Mode",
    ["Structure", "Energy", "Simulation", "Explorer"]
)

# =============================
# MAIN LOGIC
# =============================

st.title("🧬 Biomolecular Teaching Platform")

# =============================
# PDB / DNA
# =============================
if dataset in ["PDB", "DNA"]:

    folder = "data/proteins" if dataset=="PDB" else "data/dna"
    files = os.listdir(folder) if os.path.exists(folder) else []

    if not files:
        st.warning("No files found")
    else:
        file = st.selectbox("Select file", files)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("x", os.path.join(folder,file))

        atoms = list(structure.get_atoms())
        coords = np.array([a.get_coord() for a in atoms])

        st.write(f"Atoms: {len(coords)}")

        # -------- STRUCTURE --------
        if mode == "Structure":
            with open(os.path.join(folder,file)) as f:
                pdb_text = f.read()
            show_3d(pdb_text)

        # -------- ENERGY --------
        elif mode == "Energy":
            E = compute_energy(coords)
            st.metric("Total Energy", f"{E:.2f}")

        # -------- SIMULATION --------
        elif mode == "Simulation":
            st.info("🚧 Basic MD coming soon")
            if st.button("Run Step"):
                coords += np.random.normal(0,0.1,coords.shape)
                st.success("Step done")

# =============================
# LIGAND
# =============================
elif dataset == "Ligand":

    smiles = st.text_input("SMILES", "CCO")

    tokens = re.findall(r'[A-Z][a-z]?', smiles)
    mw = len(tokens)*12
    st.write("Atoms:", tokens)
    st.write("MW:", mw)

    if mode == "Energy":
        st.write("Toy energy:", len(tokens)*1.5)

# =============================
# FORCE FIELD
# =============================
elif dataset == "Force Field":

    tab = st.selectbox("FF Section", ["Atoms","Bonds","Angles","Dihedrals","Nonbonded"])

    if tab=="Atoms": st.dataframe(atoms_df)
    elif tab=="Bonds": st.dataframe(bonds_df)
    elif tab=="Angles": st.dataframe(angles_df)
    elif tab=="Dihedrals": st.dataframe(dihedrals_df)
    elif tab=="Nonbonded": st.dataframe(nonbonded_df)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🔬 Multi-Dataset Biomolecular Platform (FF + PDB + DNA + Ligands)")
