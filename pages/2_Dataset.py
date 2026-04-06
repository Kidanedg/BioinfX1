import streamlit as st
import numpy as np
import pandas as pd
import re
import tempfile
import os

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
for key in ["coords", "elements", "topology", "system"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =============================
# FILE UPLOAD
# =============================
st.sidebar.header("📂 Upload Files")

pdb_file = st.sidebar.file_uploader("Upload PDB", type=["pdb"])
prm_file = st.sidebar.file_uploader("Upload CHARMM .prm", type=["prm"])
rtf_file = st.sidebar.file_uploader("Upload CHARMM .rtf", type=["rtf"])
frcmod_file = st.sidebar.file_uploader("Upload AMBER .frcmod", type=["frcmod"])

# =============================
# 1. PARAMETER PARSER
# =============================
def parse_charmm_prm(file):
    params = {"bonds": [], "angles": [], "dihedrals": [], "vdw": []}
    
    for line in file:
        line = line.decode("utf-8").strip()
        
        if line.startswith("BOND"):
            parts = line.split()
            params["bonds"].append(parts)
        
        elif line.startswith("ANGLE"):
            params["angles"].append(line.split())
        
        elif line.startswith("DIHEDRAL"):
            params["dihedrals"].append(line.split())
        
        elif line.startswith("NONBONDED"):
            params["vdw"].append(line.split())
    
    return params

# =============================
# 2. AMBER FRCMOD PARSER
# =============================
def parse_frcmod(file):
    params = {"vdw": [], "bonds": [], "angles": []}
    
    for line in file:
        line = line.decode("utf-8").strip()
        if len(line) < 2:
            continue
        
        if "MASS" in line:
            continue
        
        if len(line.split()) > 2:
            params["vdw"].append(line.split())
    
    return params

# =============================
# 3. VAN DER WAALS ENERGY
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
            
            # Lennard-Jones
            lj = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
            energy += lj
    
    return energy

# =============================
# 4. LOAD PDB INTO OPENMM
# =============================
def load_structure(pdb_file):
    pdb = PDBFile(pdb_file)
    return pdb

# =============================
# 5. CREATE SYSTEM (CHARMM/AMBER)
# =============================
def create_system(pdb):
    forcefield = ForceField(
        'amber14-all.xml',
        'amber14/tip3pfb.xml'
    )
    
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        constraints=HBonds
    )
    
    return system

# =============================
# 6. GPU SIMULATION
# =============================
def run_simulation(pdb):
    system = create_system(pdb)
    
    integrator = LangevinIntegrator(
        300*kelvin,
        1/picosecond,
        0.002*picoseconds
    )
    
    platform = Platform.getPlatformByName("CUDA")  # GPU
    
    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform
    )
    
    simulation.context.setPositions(pdb.positions)
    
    simulation.minimizeEnergy()
    
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    
    return energy

# =============================
# 7. DOCKING (SIMPLE SCORING)
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
    st.success("✅ PDB Loaded")
    
    pdb = load_structure(pdb_file)
    
    coords = np.array([
        [atom.x, atom.y, atom.z]
        for atom in pdb.positions.value_in_unit(nanometer)
    ])
    
    st.session_state.coords = coords
    
    st.subheader("📊 Energy Calculation")
    
    if st.button("Compute van der Waals Energy"):
        vdw_energy = compute_vdw(coords)
        st.write("💥 VdW Energy:", vdw_energy)
    
    if st.button("Run OpenMM GPU Simulation"):
        energy = run_simulation(pdb)
        st.write("⚡ Potential Energy:", energy)

# =============================
# DOCKING SECTION
# =============================
st.subheader("🧬 Protein–Ligand Docking")

ligand_file = st.file_uploader("Upload Ligand PDB", type=["pdb"])

if ligand_file and pdb_file:
    protein = load_structure(pdb_file)
    ligand = load_structure(ligand_file)
    
    protein_coords = np.array([
        [a.x, a.y, a.z]
        for a in protein.positions.value_in_unit(nanometer)
    ])
    
    ligand_coords = np.array([
        [a.x, a.y, a.z]
        for a in ligand.positions.value_in_unit(nanometer)
    ])
    
    if st.button("Compute Docking Score"):
        score = docking_score(protein_coords, ligand_coords)
        st.write("🔗 Binding Score:", score)

# =============================
# PARAMETER DISPLAY
# =============================
st.subheader("📄 Parameter Files")

if prm_file:
    params = parse_charmm_prm(prm_file)
    st.write("CHARMM Parameters:", params)

if frcmod_file:
    params = parse_frcmod(frcmod_file)
    st.write("AMBER Parameters:", params)
