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

# ================= SESSION =================
coords = st.session_state.get("coords", None)
elements = st.session_state.get("elements", None)

st.title("⚙️ Simulation Engine")
st.sidebar.success(f"GPU Enabled: {GPU}")

if coords is None:
    st.warning("⚠️ Run Structure Analysis first to load molecule.")
    st.stop()

coords = xp.array(coords)  # ensure xp type
N = len(coords)

# ================= PARAMETERS =================
dt = st.sidebar.slider("Time Step", 0.001, 0.02, 0.005)
steps = st.sidebar.slider("Steps", 50, 500, 200)

# ================= FORCE FIELD =================
sigma = 3.5
epsilon = 0.2
k_bond = 200
r0 = 1.5

charges = xp.random.uniform(-0.5, 0.5, N)

# ================= FORCES =================
def compute_forces(coords):
    forces = xp.zeros_like(coords)

    # Bond forces
    for i in range(N - 1):
        rij = coords[i] - coords[i+1]
        r = xp.linalg.norm(rij) + 1e-9
        f = -2 * k_bond * (r - r0) * (rij / r)
        forces[i] += f
        forces[i+1] -= f

    # Lennard-Jones
    for i in range(N):
        for j in range(i+2, N):
            rij = coords[i] - coords[j]
            r = xp.linalg.norm(rij) + 1e-9

            f_mag = 24*epsilon*((2*(sigma**12)/r**13) - ((sigma**6)/r**7))
            f_vec = f_mag * (rij / r)

            forces[i] += f_vec
            forces[j] -= f_vec

    return forces

# ================= ENERGY =================
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

# ================= MINIMIZATION =================
def minimize(coords, iterations=100, lr=0.001):
    coords = coords.copy()

    for _ in range(iterations):
        forces = compute_forces(coords)
        coords += lr * forces

    return coords

if st.button("🧘 Energy Minimization"):
    coords = minimize(coords)
    st.session_state.coords = coords.get() if GPU else coords
    st.success("Minimized")

# ================= MD SIM =================
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

        # ✅ FIXED HERE
        if GPU:
            traj.append(cp.asnumpy(coords))
        else:
            traj.append(coords.copy())

        energies.append(compute_energy(coords))

    return traj, energies

# ================= RUN =================
if st.button("🚀 Run MD Simulation"):
    traj, energies = run_md(coords)

    st.session_state.traj = traj
    st.session_state.energies = energies

    st.success("Simulation complete")

# ================= PLOT =================
if "energies" in st.session_state:
    st.line_chart(st.session_state.energies)

# ================= VIEW =================
try:
    import py3Dmol
except:
    py3Dmol = None

if "traj" in st.session_state:
    frame = st.slider("Frame", 0, len(st.session_state.traj)-1, 0)
    current = st.session_state.traj[frame]

    if py3Dmol:
        pdb_str = ""
        for i,(x,y,z) in enumerate(current):
            pdb_str += f"ATOM {i:5d} C MOL 1 {x:8.3f}{y:8.3f}{z:8.3f}\n"

        viewer = py3Dmol.view(width=700, height=500)
        viewer.addModel(pdb_str, "pdb")
        viewer.setStyle({"stick":{}})
        viewer.zoomTo()

        st.components.v1.html(viewer._make_html(), height=500)

# ================= METRIC =================
E = compute_energy(coords)
st.metric("Energy", f"{E:.3f}")

st.success("🔥 MD Engine Running Successfully")
