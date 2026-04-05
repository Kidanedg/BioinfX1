import streamlit as st
import numpy as np
import pandas as pd
import os

# Optional 3D viewer
try:
    import py3Dmol
except:
    py3Dmol = None

st.title("⚙️ Simulation Engine")

# -----------------------------
# CHECK SESSION DATA
# -----------------------------
coords = st.session_state.get("coords", None)

if coords is None:
    st.warning("⚠️ Run Structure Analysis first to load molecule.")
    st.stop()

atom_types = ["C"] * len(coords)

# -----------------------------
# ENERGY FUNCTION (AMBER-LIKE)
# -----------------------------
def compute_energy(coords, atom_types):

    E_bond = E_angle = E_dihedral = E_vdw = E_elec = 0

    # Bond
    for i in range(len(coords)-1):
        r = np.linalg.norm(coords[i] - coords[i+1])
        r = max(r, 1e-6)
        E_bond += 100 * (r - 1.5)**2

    # Angle
    for i in range(len(coords)-2):
        v1 = coords[i] - coords[i+1]
        v2 = coords[i+2] - coords[i+1]
        denom = np.linalg.norm(v1)*np.linalg.norm(v2)
        if denom == 0:
            continue
        theta = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/denom, -1, 1)))
        E_angle += 20 * (theta - 109.5)**2

    # Dihedral
    for i in range(len(coords)-3):
        E_dihedral += 2 * (1 + np.cos(np.radians(i)))

    # Nonbonded
    for i in range(len(coords)):
        for j in range(i+2, len(coords)):
            r = np.linalg.norm(coords[i] - coords[j])
            if r < 1e-6:
                continue
            E_vdw += 4 * ((3.5/r)**12 - (3.5/r)**6)
            E_elec += (0.1 * 0.1) / r

    return {
        "Bond": E_bond,
        "Angle": E_angle,
        "Dihedral": E_dihedral,
        "vdW": E_vdw,
        "Electrostatic": E_elec,
        "Total": E_bond + E_angle + E_dihedral + E_vdw + E_elec
    }

# -----------------------------
# COMPUTE ENERGY
# -----------------------------
energy = compute_energy(coords, atom_types)

st.metric("Total Energy", f"{energy['Total']:.2f}")

# -----------------------------
# ENERGY BREAKDOWN (YOUR PART FIXED)
# -----------------------------
show_details = st.checkbox("Show Detailed Energy Breakdown")

if show_details:
    df = pd.DataFrame({
        "Energy Type": ["Bond", "Angle", "Dihedral", "vdW", "Electrostatic"],
        "Value": [
            energy["Bond"],
            energy["Angle"],
            energy["Dihedral"],
            energy["vdW"],
            energy["Electrostatic"]
        ]
    })

    st.table(df)
    st.bar_chart(df.set_index("Energy Type"))

# -----------------------------
# 🎛️ INTERACTIVE LJ SIMULATION
# -----------------------------
st.markdown("### 🎛️ Lennard-Jones Interaction")

r = st.slider("Distance (r)", 0.5, 10.0, 3.0, 0.1)
sigma = st.slider("Sigma (σ)", 1.0, 5.0, 3.5, 0.1)
epsilon = st.slider("Epsilon (ε)", 0.01, 1.0, 0.2, 0.01)

def lj(r, sigma, epsilon):
    r = max(r, 1e-6)
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

lj_energy = lj(r, sigma, epsilon)

st.metric("LJ Energy", f"{lj_energy:.4f}")

# Plot curve
r_vals = np.linspace(0.5, 10, 200)
lj_vals = [lj(rv, sigma, epsilon) for rv in r_vals]

df_lj = pd.DataFrame({"Distance": r_vals, "Energy": lj_vals})
st.line_chart(df_lj.set_index("Distance"))

# -----------------------------
# ⏱️ MOLECULAR DYNAMICS
# -----------------------------
st.markdown("### ⏱️ Molecular Dynamics")

def run_md(coords, steps=50, dt=0.01):
    coords = coords.copy()
    velocities = np.zeros_like(coords)
    traj = []

    for _ in range(steps):
        forces = np.random.randn(*coords.shape) * 0.1
        velocities += forces * dt
        coords += velocities * dt
        traj.append(coords.copy())

    return traj

if st.button("Run Simulation"):

    traj = run_md(coords, steps=100)

    energies = []
    for frame in traj:
        e = compute_energy(frame, atom_types)
        energies.append(e["Total"])

    st.line_chart(energies)

    st.session_state.traj = traj

# -----------------------------
# FRAME VIEWER
# -----------------------------
traj = st.session_state.get("traj", None)

if traj is not None:
    st.markdown("### 🎞️ Trajectory Viewer")
    frame = st.slider("Frame", 0, len(traj)-1, 0)

    current = traj[frame]
    st.write(f"Frame {frame} | Atoms: {len(current)}")

# -----------------------------
# 🎥 3D VISUALIZATION (OPTIONAL)
# -----------------------------
if py3Dmol and os.path.exists("data/proteins"):
    st.markdown("### 🎥 3D Viewer")

    files = os.listdir("data/proteins")
    if files:
        selected = st.selectbox("Select PDB", files)

        with open(os.path.join("data/proteins", selected)) as f:
            pdb = f.read()

        viewer = py3Dmol.view(width=600, height=400)
        viewer.addModel(pdb, "pdb")
        viewer.setStyle({"stick": {}})
        viewer.zoomTo()

        st.components.v1.html(viewer._make_html(), height=400)

# -----------------------------
# END
# -----------------------------
st.markdown("---")
st.success("✅ Dynamic Biomolecular Simulation Running")
