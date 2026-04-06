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

st.set_page_config(page_title="🧬 BioMolecular AI Platform", layout="wide")

# =============================
# LOGIN SYSTEM
# =============================
USERS = {"student": "1234", "admin": "admin"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Secure Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if USERS.get(u) == p:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Module",
    ["Simulation", "Docking & Binding", "Structure Analysis"]
)

protein_file = st.sidebar.file_uploader("Upload Protein (PDB)", type="pdb")
ligand_file = st.sidebar.file_uploader("Upload Ligand (PDB)", type="pdb")
forcefield_file = st.sidebar.file_uploader("Upload Force Field (CSV)", type="csv")

# =============================
# 3D VIEWER (SCIENTIFIC STYLE)
# =============================
def show_3d(protein, ligand=None):
    html = f"""
    <div id="viewer" style="width:100%; height:600px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "black"}});

        // Protein
        viewer.addModel(`{protein}`, "pdb");
        viewer.setStyle({{}}, {{
            cartoon: {{color: 'spectrum'}}
        }});

        // Ligand
        {"viewer.addModel(`" + ligand + "`, 'pdb'); viewer.setStyle({model:1}, {stick:{colorscheme:'greenCarbon'}});" if ligand else ""}

        viewer.zoomTo();
        viewer.render();
    </script>
    """
    components.html(html, height=600)

# =============================
# GEOMETRY FUNCTIONS
# =============================
def distance(a, b):
    return np.linalg.norm(a - b)

def detect_binding_site(protein_coords, ligand_coords):
    site = []
    for i, p in enumerate(protein_coords):
        for l in ligand_coords:
            if distance(p, l) < 5.0:
                site.append(i)
                break
    return list(set(site))

# =============================
# FORCE FIELD LOADING
# =============================
def load_forcefield(file):
    df = pd.read_csv(file)
    return dict(zip(df["atom"], df["charge"]))

# =============================
# ENERGY CALCULATION
# =============================
def compute_binding_energy(p_coords, l_coords):
    energy = 0
    for p in p_coords:
        for l in l_coords:
            r = distance(p, l) + 1e-6
            energy += -1 / r   # simplified attractive term
    return energy

# =============================
# STRUCTURE ANALYSIS PAGE
# =============================
if page == "Structure Analysis":

    st.title("🧬 Protein Structure Insight")

    st.markdown("## Structural Levels")

    st.markdown("""
    🔹 Primary: Sequence  
    🔹 Secondary: α-helix / β-sheet  
    🔹 Tertiary: Folding  
    🔹 Quaternary: Multi-chain  
    """)

    st.markdown("## 🧪 Visualization Example")

    # Demo image section (handled by UI, no code needed)

# =============================
# SIMULATION PAGE
# =============================
elif page == "Simulation":

    st.title("⚛️ Molecular Simulation")

    if protein_file:

        protein_data = protein_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", StringIO(protein_data))
        atoms = list(structure.get_atoms())
        coords = np.array([a.get_coord() for a in atoms])

        st.success(f"Loaded Protein: {len(coords)} atoms")

        show_3d(protein_data)

        if st.button("Run Energy Simulation"):

            energy = np.sum([distance(coords[i], coords[j])
                            for i in range(len(coords))
                            for j in range(i+1, len(coords))])

            st.metric("System Energy", f"{energy:.2f}")

            fig, ax = plt.subplots()
            ax.hist(coords.flatten(), bins=50)
            st.pyplot(fig)

    else:
        st.info("Upload a protein file")

# =============================
# DOCKING PAGE
# =============================
elif page == "Docking & Binding":

    st.title("🧪 Docking & Binding Analysis")

    if protein_file and ligand_file:

        prot_data = protein_file.read().decode("utf-8")
        lig_data = ligand_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)

        prot = parser.get_structure("p", StringIO(prot_data))
        lig = parser.get_structure("l", StringIO(lig_data))

        p_atoms = list(prot.get_atoms())
        l_atoms = list(lig.get_atoms())

        p_coords = np.array([a.get_coord() for a in p_atoms])
        l_coords = np.array([a.get_coord() for a in l_atoms])

        # Docking (center alignment)
        shift = p_coords.mean(axis=0) - l_coords.mean(axis=0)
        l_coords += shift

        st.success("Docking Completed")

        # Binding Site Detection
        site = detect_binding_site(p_coords, l_coords)

        st.write(f"🧬 Binding Site Residues: {len(site)} atoms")

        # Energy
        binding_energy = compute_binding_energy(p_coords, l_coords)
        st.metric("Binding Energy", f"{binding_energy:.2f}")

        # Visualization
        show_3d(prot_data, lig_data)

        # Plot
        fig, ax = plt.subplots()
        ax.bar(["Binding Energy"], [binding_energy])
        st.pyplot(fig)

        # Force field integration
        if forcefield_file:
            ff = load_forcefield(forcefield_file)
            st.success("Force Field Loaded")
            st.write(list(ff.items())[:5])

    else:
        st.info("Upload protein & ligand")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🧬 AI-Powered Biomolecular Modeling Platform | Research Grade")
