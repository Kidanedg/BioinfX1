import streamlit as st
import numpy as np
import pandas as pd
import re
import tempfile
import os
from io import StringIO

# =============================
# OPENMM IMPORTS
# =============================
try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
except:
    st.error("Install OpenMM in requirements.txt")
    st.stop()

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Full CHARMM Engine", layout="wide")
st.title("🧬 Full Molecular Mechanics Platform (CHARMM + AMBER + Docking)")

# =============================
# SESSION STATE
# =============================
for key in ["coords"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================
# FILE UPLOAD
# =============================
st.sidebar.header("📂 Upload Files")

pdb_file = st.sidebar.file_uploader("Upload PDB", type=["pdb"])
ligand_file = st.sidebar.file_uploader("Upload Ligand PDB", type=["pdb"])
prm_file = st.sidebar.file_uploader("Upload CHARMM .prm", type=["prm"])
frcmod_file = st.sidebar.file_uploader("Upload AMBER .frcmod", type=["frcmod"])

# =============================
# 🔥 ROBUST PDB LOADER (FIXED)
# =============================
def load_structure(uploaded_file):
    try:
        pdb_string = uploaded_file.read().decode("utf-8")

        # Keep only valid lines
        lines = [
            line for line in pdb_string.splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]

        if len(lines) == 0:
            raise ValueError("No ATOM/HETATM records found.")

        fixed_pdb = "\n".join(lines) + "\nEND\n"

        pdb = PDBFile(StringIO(fixed_pdb))
        return pdb

    except Exception as e:
        st.error(f"❌ PDB loading failed: {e}")
        return None

# =============================
# ✅ SAFE COORD EXTRACTION
# =============================
def get_coords(pdb):
    return np.array([
        [pos[0], pos[1], pos[2]]
        for pos in pdb.positions.value_in_unit(nanometer)
    ])

# =============================
# VAN DER WAALS ENERGY
# =============================
def compute_vdw(coords):
    n = len(coords)
    energy = 0.0

    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(coords[i] - coords[j])
            if r < 0.1:
                continue

            epsilon = 0.2
            sigma = 3.5

            energy += 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

    return energy

# =============================
# CREATE SYSTEM
# =============================
def create_system(pdb):
    forcefield = ForceField(
        'amber14-all.xml',
        'amber14/tip3pfb.xml'
    )

    return forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        constraints=HBonds
    )

# =============================
# GPU SIMULATION (SAFE)
# =============================
def run_simulation(pdb):
    system = create_system(pdb)

    integrator = LangevinIntegrator(
        300*kelvin,
        1/picosecond,
        0.002*picoseconds
    )

    # Try GPU → fallback CPU
    try:
        platform = Platform.getPlatformByName("CUDA")
    except:
        platform = Platform.getPlatformByName("CPU")

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform
    )

    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy()

# =============================
# DOCKING SCORE (SAFE)
# =============================
def docking_score(protein_coords, ligand_coords):
    score = 0.0

    for p in protein_coords:
        for l in ligand_coords:
            r = np.linalg.norm(p - l)

            if r < 8.0:
                score += 1 / (r + 1e-6)

    return score

# =============================
# MAIN WORKFLOW
# =============================
if pdb_file:
    pdb = load_structure(pdb_file)

    if pdb:
        st.success("✅ Protein Loaded")

        coords = get_coords(pdb)
        st.session_state.coords = coords

        st.subheader("📊 Energy Calculation")

        if st.button("Compute van der Waals Energy"):
            st.write("💥 VdW Energy:", compute_vdw(coords))

        if st.button("Run OpenMM Simulation"):
            st.write("⚡ Energy:", run_simulation(pdb))

# =============================
# DOCKING
# =============================
st.subheader("🧬 Protein–Ligand Docking")

if pdb_file and ligand_file:
    protein = load_structure(pdb_file)
    ligand = load_structure(ligand_file)

    if protein and ligand:
        protein_coords = get_coords(protein)
        ligand_coords = get_coords(ligand)

        if st.button("Compute Docking Score"):
            score = docking_score(protein_coords, ligand_coords)
            st.write("🔗 Binding Score:", score)

# =============================
# PARAMETER PARSER (IMPROVED)
# =============================
def parse_charmm_prm(file):
    params = {"bonds": [], "angles": [], "dihedrals": [], "vdw": []}

    for line in file:
        line = line.decode("utf-8").strip()

        if not line or line.startswith("*"):
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        if len(parts) == 4:
            params["bonds"].append(parts)
        elif len(parts) == 5:
            params["angles"].append(parts)
        elif len(parts) >= 6:
            params["dihedrals"].append(parts)

    return params

def parse_frcmod(file):
    params = {"vdw": []}

    for line in file:
        line = line.decode("utf-8").strip()

        if len(line.split()) > 2:
            params["vdw"].append(line.split())

    return params

# =============================
# PARAM DISPLAY
# =============================
st.subheader("📄 Parameter Files")

if prm_file:
    st.write("CHARMM:", parse_charmm_prm(prm_file))

if frcmod_file:
    st.write("AMBER:", parse_frcmod(frcmod_file))
