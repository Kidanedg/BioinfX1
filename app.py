import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="🧬 BioMolecular AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS (🔥 UI BOOST)
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}
.metric-box {
    background: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
h1, h2, h3 {
    color: #00d4ff;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SAFE IMPORT
# =============================
try:
    from Bio.PDB import PDBParser
except:
    st.error("Install Biopython: pip install biopython")
    st.stop()

# =============================
# LOGIN SYSTEM
# =============================
USERS = {"student": "1234", "admin": "admin"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("# 🔐 BioMolecular Platform Login")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("🚀 Login", use_container_width=True):
            if USERS.get(u) == p:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    st.stop()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🧭 Navigation Panel")

page = st.sidebar.radio(
    "Select Module",
    ["🧬 Structure Analysis", "⚛️ Simulation", "🧪 Docking"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Input Data")

protein_file = st.sidebar.file_uploader("Protein (PDB)", type="pdb")
ligand_file = st.sidebar.file_uploader("Ligand (PDB)", type="pdb")
forcefield_file = st.sidebar.file_uploader("Force Field (CSV)", type="csv")

# =============================
# 3D VIEWER (UPGRADED)
# =============================
def show_3d(protein, ligand=None):
    ligand_js = ""

    if ligand:
        ligand_js = f"""
        viewer.addModel(`{ligand}`, 'pdb');
        viewer.setStyle({{model:1}}, {{
            stick: {{colorscheme:'cyanCarbon'}},
            sphere: {{scale:0.3}}
        }});
        """

    html = f"""
    <div id="viewer" style="width:100%; height:650px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>

    <script>
        let viewer = $3Dmol.createViewer("viewer", {{
            backgroundColor: "black"
        }});

        viewer.addModel(`{protein}`, "pdb");

        viewer.setStyle({{}}, {{
            cartoon: {{color: 'spectrum'}}
        }});

        {ligand_js}

        viewer.addSurface($3Dmol.SurfaceType.VDW, {{
            opacity: 0.4,
            color: "white"
        }});

        viewer.setHoverable({{}}, true,
            function(atom, viewer) {{
                if (atom) {{
                    viewer.addLabel(atom.resn + " " + atom.resi, {{
                        position: atom,
                        backgroundColor: "black",
                        fontColor: "white"
                    }});
                }}
            }},
            function(atom, viewer) {{
                viewer.removeAllLabels();
            }}
        );

        viewer.zoomTo();
        viewer.render();
    </script>
    """
    components.html(html, height=650)

# =============================
# GEOMETRY
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

def compute_binding_energy(p_coords, l_coords):
    energy = 0
    for p in p_coords:
        for l in l_coords:
            r = distance(p, l) + 1e-6
            energy += -1 / r
    return energy

def load_forcefield(file):
    df = pd.read_csv(file)
    return dict(zip(df["atom"], df["charge"]))

# =============================
# STRUCTURE PAGE
# =============================
if page == "🧬 Structure Analysis":

    st.title("🧬 Protein Structure Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Primary", "Sequence")
    col2.metric("Secondary", "α / β")
    col3.metric("Tertiary", "3D Fold")
    col4.metric("Quaternary", "Multi-chain")

    st.markdown("---")

    if protein_file:
        protein_data = protein_file.read().decode("utf-8")
        show_3d(protein_data)
    else:
        st.info("Upload a protein to visualize")

# =============================
# SIMULATION
# =============================
elif page == "⚛️ Simulation":

    st.title("⚛️ Molecular Simulation")

    if protein_file:
        protein_data = protein_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", StringIO(protein_data))
        atoms = list(structure.get_atoms())
        coords = np.array([a.get_coord() for a in atoms])

        st.success(f"Loaded {len(coords)} atoms")

        show_3d(protein_data)

        if st.button("⚡ Run Simulation"):

            energy = np.sum([
                distance(coords[i], coords[j])
                for i in range(len(coords))
                for j in range(i+1, len(coords))
            ])

            c1, c2 = st.columns(2)
            c1.metric("System Energy", f"{energy:.2f}")
            c2.metric("Atoms", len(coords))

            fig, ax = plt.subplots()
            ax.hist(coords.flatten(), bins=50, color="cyan")
            ax.set_title("Atomic Distribution")
            st.pyplot(fig)

    else:
        st.warning("Upload a protein")

# =============================
# DOCKING
# =============================
elif page == "🧪 Docking":

    st.title("🧪 Docking & Binding")

    if protein_file and ligand_file:

        prot_data = protein_file.read().decode("utf-8")
        lig_data = ligand_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)

        prot = parser.get_structure("p", StringIO(prot_data))
        lig = parser.get_structure("l", StringIO(lig_data))

        p_coords = np.array([a.get_coord() for a in prot.get_atoms()])
        l_coords = np.array([a.get_coord() for a in lig.get_atoms()])

        # Docking
        shift = p_coords.mean(axis=0) - l_coords.mean(axis=0)
        l_coords += shift

        site = detect_binding_site(p_coords, l_coords)
        energy = compute_binding_energy(p_coords, l_coords)

        st.success("Docking Completed")

        c1, c2, c3 = st.columns(3)
        c1.metric("Binding Energy", f"{energy:.2f}")
        c2.metric("Binding Atoms", len(site))
        c3.metric("Ligand Atoms", len(l_coords))

        show_3d(prot_data, lig_data)

        fig, ax = plt.subplots()
        ax.bar(["Energy"], [energy], color="lime")
        st.pyplot(fig)

        if forcefield_file:
            ff = load_forcefield(forcefield_file)
            st.success("Force Field Loaded")
            st.dataframe(pd.DataFrame(list(ff.items()), columns=["Atom", "Charge"]))

    else:
        st.info("Upload protein & ligand")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("🚀 AI Biomolecular Platform | Built for Research & Discovery")
