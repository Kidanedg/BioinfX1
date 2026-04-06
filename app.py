import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# =============================
# ⚙️ CONFIG
# =============================
st.set_page_config(
    page_title="🧬 BioMolecular AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 🎨 ADVANCED UI THEME
# =============================
st.markdown("""
<style>

/* ===== GLOBAL ===== */
body {
    background: linear-gradient(135deg, #0e1117, #1c1f26);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== HEADERS ===== */
h1, h2, h3 {
    color: #00d4ff;
    font-weight: 600;
}

/* ===== METRIC BOX ===== */
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(6px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

/* ===== LOGIN CARD ===== */
.login-card {
    background: rgba(255,255,255,0.06);
    padding: 35px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 30px rgba(0,0,0,0.4);
    text-align: center;
}

/* ===== INPUT FIELDS ===== */
.stTextInput input {
    border-radius: 10px;
    padding: 10px;
}

/* ===== BUTTON ===== */
.stButton button {
    border-radius: 12px;
    background: linear-gradient(90deg, #00d4ff, #0072ff);
    color: white;
    font-weight: bold;
    padding: 10px;
    transition: 0.3s;
}
.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #0072ff, #00d4ff);
}

/* ===== FOOT NOTE ===== */
.small-text {
    font-size: 12px;
    color: #aaaaaa;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =============================
# 🔬 SAFE IMPORT
# =============================
try:
    from Bio.PDB import PDBParser
except:
    st.error("⚠️ Install Biopython: pip install biopython")
    st.stop()

# =============================
# 🔐 LOGIN SYSTEM (ENHANCED)
# =============================
USERS = {"student": "1234", "admin": "admin"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("## 🧬 BioMolecular AI Platform")
    st.caption("Secure access to simulation, docking, and molecular analytics")

    # Centered layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        st.markdown("### 🔐 Login Portal")

        u = st.text_input("👤 Username")
        p = st.text_input("🔑 Password", type="password")

        login_btn = st.button("🚀 Login", use_container_width=True)

        if login_btn:
            if USERS.get(u) == p:
                st.session_state.logged_in = True
                st.success("✅ Access granted")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

        st.markdown(
            '<div class="small-text">© 2026 Kidane Desta • Aksum University (AkU)</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# =============================
# 🧭 SIDEBAR (ENHANCED)
# =============================
with st.sidebar:

    st.markdown("## 🧭 Navigation")
    st.caption("Explore molecular modules")

    page = st.radio(
        "Select Module",
        ["🧬 Structure Analysis", "⚛️ Simulation", "🧪 Docking"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # =============================
    # 📂 DATA INPUT SECTION
    # =============================
    st.markdown("### 📂 Input Data")

    protein_file = st.file_uploader("🧬 Protein (PDB)", type="pdb")
    ligand_file = st.file_uploader("🧪 Ligand (PDB)", type="pdb")
    forcefield_file = st.file_uploader("⚛️ Force Field (CSV)", type="csv")

    st.markdown("---")

    # =============================
    # ⚙️ VIEWER SETTINGS
    # =============================
    st.markdown("### ⚙️ Viewer Settings")

    show_surface = st.toggle("Surface", value=True)
    show_labels = st.toggle("Atom Labels", value=True)
    style = st.selectbox("Style", ["Cartoon", "Stick", "Sphere"])

    st.markdown("---")

    st.caption("© 2026 Kidane Desta • AkU")


# =============================
# 🧊 3D VIEWER (ENHANCED)
# =============================
def show_3d(protein, ligand=None):

    # =============================
    # 🎨 STYLE CONTROL
    # =============================
    if style == "Cartoon":
        style_js = "cartoon: {color: 'spectrum'}"
    elif style == "Stick":
        style_js = "stick: {}"
    else:
        style_js = "sphere: {scale: 0.3}"

    ligand_js = ""

    if ligand:
        ligand_js = f"""
        viewer.addModel(`{ligand}`, 'pdb');
        viewer.setStyle({{model:1}}, {{
            stick: {{colorscheme:'cyanCarbon'}},
            sphere: {{scale:0.3}}
        }});
        """

    surface_js = ""
    if show_surface:
        surface_js = """
        viewer.addSurface($3Dmol.SurfaceType.VDW, {
            opacity: 0.35,
            color: "white"
        });
        """

    label_js = ""
    if show_labels:
        label_js = """
        viewer.setHoverable({}, true,
            function(atom, viewer) {
                if (atom) {
                    viewer.addLabel(atom.elem + " (" + atom.resn + atom.resi + ")", {
                        position: atom,
                        backgroundColor: "black",
                        fontColor: "white",
                        fontSize: 12
                    });
                }
            },
            function(atom, viewer) {
                viewer.removeAllLabels();
            }
        );
        """

    # =============================
    # 🌐 HTML VIEWER
    # =============================
    html = f"""
    <div id="viewer" style="width:100%; height:700px; border-radius:12px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>

    <script>
        let viewer = $3Dmol.createViewer("viewer", {{
            backgroundColor: "#0e1117"
        }});

        viewer.addModel(`{protein}`, "pdb");

        viewer.setStyle({{}}, {{
            {style_js}
        }});

        {ligand_js}
        {surface_js}
        {label_js}

        viewer.zoomTo();
        viewer.render();
    </script>
    """

    components.html(html, height=700)
# =============================
# GEOMETRY (OPTIMIZED + CLEAN)
# =============================
def distance(a, b):
    """Euclidean distance between two 3D points"""
    return np.linalg.norm(a - b)


def detect_binding_site(protein_coords, ligand_coords, cutoff=5.0):
    """
    Detect binding site atoms within cutoff distance
    Vectorized for performance
    """
    dists = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis=2
    )
    site_indices = np.where((dists < cutoff).any(axis=1))[0]
    return site_indices.tolist()


def compute_binding_energy(p_coords, l_coords):
    """
    Simplified Lennard-Jones–like attractive term
    """
    dists = np.linalg.norm(
        p_coords[:, None, :] - l_coords[None, :, :],
        axis=2
    ) + 1e-6

    energy = np.sum(-1.0 / dists)
    return energy


def load_forcefield(file):
    """Load atom charge parameters"""
    df = pd.read_csv(file)
    return dict(zip(df["atom"], df["charge"]))


# =============================
# STRUCTURE PAGE (ENHANCED UI)
# =============================
if page == "🧬 Structure Analysis":

    st.title("🧬 Protein Structure Intelligence")

    st.markdown(
        "Explore hierarchical organization of biomolecules inspired by "
        "modern "
    )

    # =============================
    # METRICS DASHBOARD
    # =============================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🧾 Primary")
        st.caption("Amino acid sequence")
        st.metric("Level", "1°")

    with col2:
        st.markdown("### 🌀 Secondary")
        st.caption("α-helices / β-sheets")
        st.metric("Level", "2°")

    with col3:
        st.markdown("### 🧩 Tertiary")
        st.caption("3D folding structure")
        st.metric("Level", "3°")

    with col4:
        st.markdown("### 🧬 Quaternary")
        st.caption("Multi-chain assembly")
        st.metric("Level", "4°")

    st.markdown("---")

    # =============================
    # VISUALIZATION PANEL
    # =============================
    if protein_file:
        protein_data = protein_file.read().decode("utf-8")

        st.success("✅ Protein structure loaded successfully")

        # Layout split
        left, right = st.columns([2, 1])

        with left:
            show_3d(protein_data)

        with right:
            st.markdown("### 🔬 Structure Insights")

            st.info(
                "• Cartoon model shows secondary structure\n"
                "• Surface reveals solvent accessibility\n"
                "• Colors represent residue progression"
            )

            st.markdown("### ⚙️ Visualization Controls")

            style = st.selectbox(
                "Rendering Style",
                ["Cartoon", "Surface", "Stick"]
            )

            color = st.selectbox(
                "Color Scheme",
                ["Spectrum", "Chain", "Hydrophobicity"]
            )

            st.markdown("---")

            st.markdown("### 📊 Quick Stats")

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("prot", StringIO(protein_data))
            atoms = list(structure.get_atoms())

            st.metric("Atoms", len(atoms))
            st.metric("Chains", len(list(structure.get_chains())))

    else:
        st.markdown("### 📂 Upload Required")
        st.warning("Please upload a protein file to begin analysis")
# =============================
# ⚛️ SIMULATION (PREMIUM UI - FIXED)
# =============================
elif page == "⚛️ Simulation":

    # =============================
    # 🎨 SECTION HEADER
    # =============================
    st.markdown("""
    <div style="
        padding:20px;
        border-radius:15px;
        background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
        color:white;
        text-align:center;
        box-shadow:0px 4px 20px rgba(0,0,0,0.4);
    ">
        <h2>⚛️ Molecular Simulation Lab</h2>
        <p>Analyze structure • Compute energy • Explore atomic interactions</p>
    </div>
    """, unsafe_allow_html=True)

    if protein_file:
        protein_data = protein_file.read().decode("utf-8")

        # =============================
        # 🔬 STRUCTURE PARSING
        # =============================
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", StringIO(protein_data))
        atoms = list(structure.get_atoms())

        if len(atoms) == 0:
            st.error("❌ No atoms found in file")
            st.stop()

        coords = np.array([a.get_coord() for a in atoms])
        n_atoms = int(len(coords))

        # =============================
        # 📊 DASHBOARD
        # =============================
        st.markdown("### 📊 System Overview")

        c1, c2, c3 = st.columns(3)

        def card(title, value):
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.06);
                padding:20px;
                border-radius:15px;
                text-align:center;
                backdrop-filter: blur(10px);
                box-shadow:0px 4px 15px rgba(0,0,0,0.3);
            ">
                <h4 style="color:#00d4ff;">{title}</h4>
                <h2>{value}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c1:
            card("🧬 Total Atoms", n_atoms)
        with c2:
            card("📏 Dimensions", coords.shape)
        with c3:
            card("📍 Mean Position", f"{np.mean(coords):.2f}")

        st.success("✅ Protein structure loaded successfully")

        # =============================
        # 🧊 VIEWER
        # =============================
        st.markdown("### 🧊 3D Molecular Structure")
        show_3d(protein_data)

        # =============================
        # ⚙️ CONTROLS PANEL
        # =============================
        st.markdown("### ⚙️ Simulation Control Panel")

        box1, box2 = st.columns([2,1])

        with box1:
            # =============================
            # 🔧 FIXED SLIDER (NO CRASH)
            # =============================
            if n_atoms < 10:
                st.error("❌ Protein too small for simulation")
                st.stop()

            min_atoms = 10 if n_atoms < 100 else 100
            max_atoms = n_atoms
            default_atoms = min(1000, n_atoms)

            sample_size = st.slider(
                "🔬 Sample Size (Performance Control)",
                min_value=min_atoms,
                max_value=max_atoms,
                value=default_atoms,
                step=1,
                help=f"Total atoms: {n_atoms} | Reduce for faster simulation"
            )

            method = st.radio(
                "⚛️ Energy Model",
                ["Distance Sum", "Lennard-Jones (approx)"],
                horizontal=True
            )

        with box2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("⚡ Run Simulation", use_container_width=True)

        # =============================
        # ⚡ SIMULATION EXECUTION
        # =============================
        if run_btn:

            progress = st.progress(0)

            with st.spinner("🧠 Running molecular computation..."):

                # ✅ Better random sampling
                coords_sample = coords[np.random.choice(n_atoms, sample_size, replace=False)]

                # Fake progress (UX)
                for i in range(100):
                    progress.progress(i + 1)

                # =============================
                # ⚛️ ENERGY CALCULATION
                # =============================
                if method == "Distance Sum":
                    energy = np.sum([
                        np.linalg.norm(coords_sample[i] - coords_sample[j])
                        for i in range(len(coords_sample))
                        for j in range(i + 1, len(coords_sample))
                    ])

                elif method == "Lennard-Jones (approx)":
                    energy = 0.0
                    for i in range(len(coords_sample)):
                        for j in range(i + 1, len(coords_sample)):
                            r = np.linalg.norm(coords_sample[i] - coords_sample[j])
                            if r > 0:
                                energy += (1 / r**12 - 2 / r**6)

                progress.empty()

                # =============================
                # 📈 RESULTS
                # =============================
                st.markdown("### 📈 Simulation Results")

                r1, r2 = st.columns(2)
                r1.metric("⚡ System Energy", f"{energy:.3f}")
                r2.metric("🔬 Sample Size", sample_size)

                st.success("🎉 Simulation completed successfully!")

                # =============================
                # 📊 DISTRIBUTION
                # =============================
                st.markdown("### 📊 Atomic Distribution")

                fig, ax = plt.subplots()
                ax.hist(coords_sample.flatten(), bins=50)
                ax.set_title("Atomic Coordinate Distribution")
                ax.set_xlabel("Coordinate Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                # =============================
                # 🔥 HEATMAP
                # =============================
                st.markdown("### 🔥 Distance Interaction Map")

                dist_matrix = np.linalg.norm(
                    coords_sample[:, None, :] - coords_sample[None, :, :],
                    axis=-1
                )

                fig2, ax2 = plt.subplots()
                im = ax2.imshow(dist_matrix)
                ax2.set_title("Pairwise Distance Matrix")
                plt.colorbar(im, ax=ax2)
                st.pyplot(fig2)

    else:
        st.warning("⚠️ Upload a protein PDB file to start simulation")
# 🧪 DOCKING (ENHANCED)
# =============================
elif page == "🧪 Docking":

    st.markdown("## 🧪 Molecular Docking Studio")
    st.caption("Protein–Ligand interaction analysis and binding evaluation")

    if protein_file and ligand_file:

        prot_data = protein_file.read().decode("utf-8")
        lig_data = ligand_file.read().decode("utf-8")

        parser = PDBParser(QUIET=True)

        prot = parser.get_structure("protein", StringIO(prot_data))
        lig = parser.get_structure("ligand", StringIO(lig_data))

        p_coords = np.array([a.get_coord() for a in prot.get_atoms()])
        l_coords = np.array([a.get_coord() for a in lig.get_atoms()])

        # =============================
        # 📊 INITIAL SUMMARY
        # =============================
        s1, s2 = st.columns(2)
        s1.metric("🧬 Protein Atoms", len(p_coords))
        s2.metric("🧪 Ligand Atoms", len(l_coords))

        st.markdown("### 🧊 Pre-Docking Visualization")
        show_3d(prot_data, lig_data)

        # =============================
        # ⚙️ DOCKING SETTINGS
        # =============================
        st.markdown("### ⚙️ Docking Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Docking Method", ["Centroid Align", "Random Search"])
        with col2:
            samples = st.slider("Search Iterations", 10, 500, 100)
        with col3:
            run_btn = st.button("🚀 Run Docking")

        # =============================
        # 🚀 RUN DOCKING
        # =============================
        if run_btn:

            with st.spinner("Running docking simulation..."):

                # -----------------------------
                # BASIC DOCKING STRATEGY
                # -----------------------------
                if method == "Centroid Align":
                    shift = p_coords.mean(axis=0) - l_coords.mean(axis=0)
                    l_coords_shifted = l_coords + shift

                elif method == "Random Search":
                    best_energy = float("inf")
                    l_coords_shifted = l_coords.copy()

                    for _ in range(samples):
                        random_shift = np.random.uniform(-5, 5, size=3)
                        trial_coords = l_coords + random_shift
                        e = compute_binding_energy(p_coords, trial_coords)

                        if e < best_energy:
                            best_energy = e
                            l_coords_shifted = trial_coords

                # -----------------------------
                # ANALYSIS
                # -----------------------------
                site = detect_binding_site(p_coords, l_coords_shifted)
                energy = compute_binding_energy(p_coords, l_coords_shifted)

                st.success("✅ Docking Completed Successfully")

                # =============================
                # 📈 RESULTS DASHBOARD
                # =============================
                r1, r2, r3 = st.columns(3)
                r1.metric("⚡ Binding Energy", f"{energy:.3f}")
                r2.metric("🔗 Binding Site Atoms", len(site))
                r3.metric("🧪 Ligand Atoms", len(l_coords_shifted))

                # =============================
                # 🧊 POST-DOCKING VISUALIZATION
                # =============================
                st.markdown("### 🧊 Docked Complex")
                show_3d(prot_data, lig_data)

                # =============================
                # 📊 ENERGY VISUALIZATION
                # =============================
                st.markdown("### 📊 Binding Energy Profile")

                fig, ax = plt.subplots()
                ax.bar(["Binding Energy"], [energy])
                ax.set_ylabel("Energy")
                ax.set_title("Docking Score")
                st.pyplot(fig)

                # =============================
                # 🔥 DISTANCE ANALYSIS
                # =============================
                st.markdown("### 🔍 Protein–Ligand Distance Map")

                dist_matrix = np.linalg.norm(
                    p_coords[:, None, :] - l_coords_shifted[None, :, :],
                    axis=-1
                )

                fig2, ax2 = plt.subplots()
                im = ax2.imshow(dist_matrix)
                ax2.set_title("Interaction Distance Matrix")
                plt.colorbar(im, ax=ax2)
                st.pyplot(fig2)

                # =============================
                # ⚛️ FORCE FIELD
                # =============================
                if forcefield_file:
                    ff = load_forcefield(forcefield_file)

                    st.markdown("### ⚛️ Force Field Parameters")
                    st.success("Force Field Loaded")

                    df_ff = pd.DataFrame(list(ff.items()), columns=["Atom Type", "Charge"])
                    st.dataframe(df_ff, use_container_width=True)

    else:
        st.info("⬆️ Upload both protein and ligand files to begin docking")
# =============================
# 🧾 FOOTER (ENHANCED)
# =============================
st.markdown("""
<style>
.footer {
    position: relative;
    margin-top: 50px;
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: #e0e0e0;
    text-align: center;
    font-size: 14px;
    box-shadow: 0px -2px 15px rgba(0,0,0,0.3);
}

.footer-title {
    font-size: 16px;
    font-weight: bold;
    color: #ffffff;
}

.footer-sub {
    font-size: 13px;
    color: #b0c4de;
}

.footer-copy {
    margin-top: 8px;
    font-size: 12px;
    color: #aaaaaa;
}
</style>

<div class="footer">
    <div class="footer-title">🚀 AI Biomolecular Platform</div>
    <div class="footer-sub">Built for Research • Simulation • Discovery</div>
    <div class="footer-copy">
        © 2026 Kidane Desta, Aksum University (AkU) • All Rights Reserved
    </div>
</div>
""", unsafe_allow_html=True)
