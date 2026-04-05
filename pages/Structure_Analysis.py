import streamlit as st
import numpy as np
import pandas as pd
import py3Dmol

# =========================================================
# 🔬 SCIENTIFIC LIBRARIES
# =========================================================
from Bio.PDB import PDBParser, DSSP
from pdbfixer import PDBFixer
from openmm.app import *
from openmm import *
from openmm.unit import *

# =========================================================
# 🔥 SESSION STATE INITIALIZATION
# =========================================================
default_state = {
    "coords": None,
    "elements": None,
    "topology": None,
    "structure": None,
    "pdb_file": None,
    "fixer": None,
    "system": None,
    "simulation": None,
    "energies": None
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# 🎯 APP TITLE
# =========================================================
st.title("🧬 Molecular Structure Analyzer (Advanced)")

# =========================================================
# 📂 FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload PDB file", type=["pdb"])

if uploaded_file is not None:
    st.session_state.pdb_file = uploaded_file

    with open("temp.pdb", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDB file uploaded successfully!")

# =========================================================
# 🧪 PARSE STRUCTURE
# =========================================================
if st.session_state.pdb_file is not None:

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("protein", "temp.pdb")
        st.session_state.structure = structure

        coords = []
        elements = []

        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
            elements.append(atom.element)

        st.session_state.coords = np.array(coords)
        st.session_state.elements = elements

        st.success(f"Structure parsed! Total atoms: {len(coords)}")

    except Exception as e:
        st.error(f"Parsing error: {e}")

# =========================================================
# 🧬 3D VIEWER
# =========================================================
st.subheader("🧬 3D Molecular Viewer")

if st.session_state.pdb_file is not None:
    with open("temp.pdb", "r") as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"stick": {}})
    view.zoomTo()

    st.components.v1.html(view._make_html(), height=500)

# =========================================================
# 🔧 FIX STRUCTURE
# =========================================================
if st.button("Fix Structure"):

    try:
        fixer = PDBFixer(filename="temp.pdb")

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.0)

        st.session_state.fixer = fixer
        st.session_state.topology = fixer.topology

        st.success("Structure fixed!")

    except Exception as e:
        st.error(f"Fix error: {e}")

# =========================================================
# ⚡ BUILD SYSTEM
# =========================================================
if st.button("Build Simulation System"):

    try:
        if st.session_state.fixer is None:
            st.warning("Fix structure first!")
        else:
            forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

            system = forcefield.createSystem(
                st.session_state.fixer.topology,
                nonbondedMethod=NoCutoff,
                constraints=HBonds
            )

            st.session_state.system = system
            st.success("System built!")

    except Exception as e:
        st.error(f"System error: {e}")

# =========================================================
# ▶️ RUN SIMULATION + ENERGY TRACKING
# =========================================================
if st.button("Run Energy Minimization"):

    try:
        if st.session_state.system is None:
            st.warning("Build system first!")
        else:
            integrator = LangevinIntegrator(
                300*kelvin,
                1/picosecond,
                0.002*picoseconds
            )

            simulation = Simulation(
                st.session_state.fixer.topology,
                st.session_state.system,
                integrator
            )

            simulation.context.setPositions(
                st.session_state.fixer.positions
            )

            energies = []

            for i in range(50):
                simulation.step(10)
                state = simulation.context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
                energies.append(energy)

            st.session_state.simulation = simulation
            st.session_state.energies = energies

            st.success("Simulation completed!")

    except Exception as e:
        st.error(f"Simulation error: {e}")

# =========================================================
# 📉 ENERGY PLOT
# =========================================================
if st.session_state.energies is not None:

    st.subheader("📉 Energy vs Step")

    df_energy = pd.DataFrame({
        "Step": range(len(st.session_state.energies)),
        "Energy": st.session_state.energies
    })

    st.line_chart(df_energy.set_index("Step"))

# =========================================================
# 🧪 FORCE FIELD INSPECTION
# =========================================================
st.subheader("🧪 Force Field Parameters")

if st.session_state.system is not None:

    system = st.session_state.system

    st.write(f"Particles: {system.getNumParticles()}")
    st.write(f"Forces: {system.getNumForces()}")

    for i in range(system.getNumForces()):
        force = system.getForce(i)
        st.write(f"🔹 {i}: {type(force).__name__}")

        if isinstance(force, HarmonicBondForce):
            st.write(f"  Bonds: {force.getNumBonds()}")

        elif isinstance(force, HarmonicAngleForce):
            st.write(f"  Angles: {force.getNumAngles()}")

        elif isinstance(force, PeriodicTorsionForce):
            st.write(f"  Torsions: {force.getNumTorsions()}")

        elif isinstance(force, NonbondedForce):
            st.write(f"  Nonbonded particles: {force.getNumParticles()}")

# =========================================================
# 📊 DSSP SECONDARY STRUCTURE
# =========================================================
st.subheader("📊 DSSP Secondary Structure")

if st.session_state.structure is not None:

    try:
        model = st.session_state.structure[0]
        dssp = DSSP(model, "temp.pdb")

        data = []
        for key in dssp.keys():
            res = dssp[key]
            data.append({
                "Chain": key[0],
                "Residue": key[1][1],
                "AA": res[1],
                "Structure": res[2]
            })

        df = pd.DataFrame(data)

        st.dataframe(df)

        st.write("Structure distribution:")
        st.write(df["Structure"].value_counts())

    except Exception:
        st.warning("DSSP not installed. Install with: conda install -c salilab dssp")

# =========================================================
# 📊 BASIC COORDINATE ANALYSIS
# =========================================================
if st.session_state.coords is not None:

    st.subheader("📊 Coordinate Summary")

    df = pd.DataFrame(
        st.session_state.coords,
        columns=["X", "Y", "Z"]
    )

    st.dataframe(df.head())
    st.write(df.describe())
