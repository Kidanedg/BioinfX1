import streamlit as st
import numpy as np
import pandas as pd

# ================= GPU SUPPORT =================
try:
    import cupy as cp
    xp = cp
    GPU = True
except:
    xp = np
    GPU = False

# ================= 3D VIEWER =================
try:
    import py3Dmol
except:
    py3Dmol = None

st.title("⚙️ Advanced Biomolecular Simulation Engine")

st.sidebar.success(f"GPU Enabled: {GPU}")

# =========================================================
# 📂 PDB UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload PDB File", type=["pdb"])

def parse_pdb(file):
    coords = []
    atoms = []
    residues = []

    for line in file.readlines():
        line = line.decode("utf-8")

        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            atom = line[12:16].strip()
            res = line[17:20].strip()

            coords.append([x, y, z])
            atoms.append(atom)
            residues.append(res)

    return xp.array(coords), atoms, residues

if uploaded_file:
    coords, atoms, residues = parse_pdb(uploaded_file)
    st.session_state.coords = coords
    st.session_state.atoms = atoms
    st.session_state.residues = residues
    st.success(f"Loaded {len(coords)} atoms")

coords = st.session_state.get("coords", None)

if coords is None:
    st.warning("Upload a PDB file to start.")
    st.stop()

coords = coords.copy()
N = len(coords)

# =========================================================
# ⚙️ PARAMETERS
# =========================================================
dt = st.sidebar.slider("Time Step", 0.001, 0.02, 0.005)
steps = st.sidebar.slider("Steps", 50, 500, 200)
temperature = st.sidebar.slider("Temperature", 50, 1000, 300)

# =========================================================
# ⚡ PARAMETERS (AMBER-LIKE SIMPLIFIED)
# =========================================================
sigma = 3.5
epsilon = 0.2
k_bond = 200
r0 = 1.5
k_angle = 40
theta0 = xp.pi
charges = xp.random.uniform(-0.5, 0.5, N)

# =========================================================
# 🔬 ENERGY + FORCES
# =========================================================
def compute_forces(coords):
    forces = xp.zeros_like(coords)

    # Bond
    for i in range(N - 1):
        rij = coords[i] - coords[i+1]
        r = xp.linalg.norm(rij) + 1e-9
        f = -2 * k_bond * (r - r0) * (rij / r)
        forces[i] += f
        forces[i+1] -= f

    # vdW (LJ)
    for i in range(N):
        for j in range(i+2, N):
            rij = coords[i] - coords[j]
            r = xp.linalg.norm(rij) + 1e-9

            f_mag = 24*epsilon*((2*(sigma**12)/r**13) - ((sigma**6)/r**7))
            f_vec = f_mag * (rij / r)

            forces[i] += f_vec
            forces[j] -= f_vec

    # Electrostatic
    for i in range(N):
        for j in range(i+1, N):
            rij = coords[i] - coords[j]
            r = xp.linalg.norm(rij) + 1e-9

            f_mag = charges[i]*charges[j]/r**2
            f_vec = f_mag * (rij / r)

            forces[i] += f_vec
            forces[j] -= f_vec

    return forces

def compute_energy(coords):
    E = 0

    for i in range(N-1):
        r = xp.linalg.norm(coords[i] - coords[i+1])
        E += k_bond * (r - r0)**2

    for i in range(N):
        for j in range(i+2, N):
            r = xp.linalg.norm(coords[i] - coords[j])
            E += 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

    return float(E)

# =========================================================
# 🧘 ENERGY MINIMIZATION
# =========================================================
def minimize(coords, iterations=100, lr=0.001):
    coords = coords.copy()

    for _ in range(iterations):
        forces = compute_forces(coords)
        coords += lr * forces  # gradient descent

    return coords

if st.button("🧘 Energy Minimization"):
    coords = minimize(coords)
    st.session_state.coords = coords
    st.success("Minimization complete")

# =========================================================
# 🚀 MD SIMULATION
# =========================================================
def run_md(coords):
    coords = coords.copy()
    v = xp.random.randn(N,3) * 0.1
    forces = compute_forces(coords)

    traj = []
    energies = []

    for _ in range(steps):
        coords += v*dt + 0.5*forces*dt**2
        new_forces = compute_forces(coords)
        v += 0.5*(forces + new_forces)*dt
        forces = new_forces

        traj.append(xp.asnumpy(coords))
        energies.append(compute_energy(coords))

    return traj, energies

if st.button("🚀 Run MD Simulation"):
    traj, energies = run_md(coords)

    st.session_state.traj = traj
    st.session_state.energies = energies

    st.success("Simulation complete")

# =========================================================
# 📈 ENERGY PLOT
# =========================================================
if "energies" in st.session_state:
    st.line_chart(st.session_state.energies)

# =========================================================
# 🎞️ VIEWER
# =========================================================
if "traj" in st.session_state:

    frame = st.slider("Frame", 0, len(st.session_state.traj)-1, 0)
    current = st.session_state.traj[frame]

    if py3Dmol:
        pdb_str = ""
        for i,(x,y,z) in enumerate(current):
            pdb_str += f"ATOM {i:5d} C MOL 1 {x:8.3f}{y:8.3f}{z:8.3f} 1.00 0.00 C\n"

        viewer = py3Dmol.view(width=700, height=500)
        viewer.addModel(pdb_str, "pdb")
        viewer.setStyle({"cartoon":{}})  # protein style
        viewer.zoomTo()

        st.components.v1.html(viewer._make_html(), height=500)

# =========================================================
# 💾 EXPORT TRAJECTORY
# =========================================================
def export_xyz(traj):
    lines = []
    for frame in traj:
        lines.append(str(len(frame)))
        lines.append("Frame")
        for x,y,z in frame:
            lines.append(f"C {x:.3f} {y:.3f} {z:.3f}")
    return "\n".join(lines)

def export_pdb(traj):
    lines = []
    for f, frame in enumerate(traj):
        for i,(x,y,z) in enumerate(frame):
            lines.append(f"ATOM {i:5d} C MOL {f:4d} {x:8.3f}{y:8.3f}{z:8.3f}")
        lines.append("ENDMDL")
    return "\n".join(lines)

if "traj" in st.session_state:
    xyz_data = export_xyz(st.session_state.traj)
    pdb_data = export_pdb(st.session_state.traj)

    st.download_button("Download XYZ", xyz_data, "trajectory.xyz")
    st.download_button("Download PDB", pdb_data, "trajectory.pdb")

# =========================================================
# 🔬 ENERGY SNAPSHOT
# =========================================================
E = compute_energy(coords)
st.metric("Energy", f"{E:.3f}")

st.markdown("---")
st.success("🔥 Research-Level MD Engine Ready")
