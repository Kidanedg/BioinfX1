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
# LOAD FORCE FIELD CSV
# -----------------------------
DATA_PATH = "data/forcefield_dataset.csv"

@st.cache_data
def load_forcefield(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return None

ff_df = load_forcefield(DATA_PATH)

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
# LIGAND MODEL
# -----------------------------
def ligand_properties(smiles):
    weights = {"C":12,"H":1,"O":16,"N":14,"S":32}
    tokens = re.findall(r'[A-Z][a-z]?', smiles)
    mw = sum(weights.get(t, 0) for t in tokens)
    logp = len(tokens) * 0.15
    return mw, logp

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
    vdw = elec = hbond = hydrophobic = 0
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
# MAIN TITLE
# -----------------------------
st.title("🧬 Biomolecular Teaching Platform (AMBER + Docking + LMS)")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.header("📂 Navigation")
page = st.sidebar.radio("Go to", ["Structure Analysis", "Force Field Explorer"])

# =========================================================
# PAGE 1: STRUCTURE ANALYSIS (YOUR ORIGINAL)
# =========================================================
if page == "Structure Analysis":

    st.sidebar.header("⚙️ Structure Config")
    data_type = st.sidebar.selectbox("Dataset Type", ["DNA", "Protein"])
    data_path = "data/dna" if data_type == "DNA" else "data/proteins"

    files = safe_listdir(data_path)

    if not files:
        st.error("No PDB files found")
        st.stop()

    selected_file = st.sidebar.selectbox("Select Structure", files)

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
    # LIGAND
    # -----------------------------
    st.subheader("💊 Ligand")
    smiles = st.text_input("Enter SMILES", "CCO")
    mw, logp = ligand_properties(smiles)

    st.write(f"MW: {mw:.2f} | LogP: {logp:.2f}")

    # -----------------------------
    # ENERGY
    # -----------------------------
    st.subheader("⚙️ Energy Calculation")

    bond_energy = angle_energy = vdw_energy = elec_energy = 0

    kb, r0 = 300, 1.5
    k_theta, theta0 = 40, 109.5
    epsilon, sigma = 0.2, 3.5

    if len(coords) > 300:
        coords = coords[np.random.choice(len(coords), 300, replace=False)]

    for i in range(len(coords)-1):
        bond_energy += kb * (distance(coords[i], coords[i+1]) - r0)**2

    for i in range(len(coords)-2):
        angle_energy += k_theta * (angle(coords[i], coords[i+1], coords[i+2]) - theta0)**2

    for i in range(len(coords)):
        for j in range(i+2, len(coords)):
            r = distance(coords[i], coords[j])
            if r < 8:
                vdw_energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
                elec_energy += (-1) / r

    total_energy = bond_energy + angle_energy + vdw_energy + elec_energy

    st.metric("Total Energy", f"{total_energy:.2f}")

    # -----------------------------
    # 3D VIEW
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

# =========================================================
# PAGE 2: FORCE FIELD EXPLORER (NEW)
# =========================================================
elif page == "Force Field Explorer":

    st.subheader("🧪 Force Field Dataset")

    if ff_df is None:
        st.error("CSV not found: data/forcefield_dataset.csv")
        st.stop()

    section = st.selectbox("Select Section", ff_df["section"].dropna().unique())

    filtered = ff_df[ff_df["section"] == section]

    st.dataframe(filtered, use_container_width=True)

    # Search
    col = st.selectbox("Search Column", ff_df.columns)
    val = st.text_input("Search Value")

    if val:
        filtered = filtered[
            filtered[col].astype(str).str.contains(val, case=False, na=False)
        ]
        st.dataframe(filtered)

    # Charts
    if section == "ATOM":
        st.bar_chart(filtered[["mass", "charge"]])

    elif section == "BOND":
        st.bar_chart(filtered[["k"]])

    elif section == "DOCKING":
        st.bar_chart(filtered.set_index("term")["weight"])

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("🔬 Integrated Platform: Structure + Force Field + Docking + LMS")
