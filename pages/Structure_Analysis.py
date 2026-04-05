import streamlit as st
import numpy as np
import pandas as pd
import py3Dmol

# =========================================================
# 🔬 SAFE IMPORTS
# =========================================================
from Bio.PDB import PDBParser, DSSP

# Optional import (CRITICAL FIX)
try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ModuleNotFoundError:
    PDBFIXER_AVAILABLE = False

# OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *

# =========================================================
# 🔥 SESSION STATE
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
# 🎯 TITLE
# =========================================================
st.title("🧬 Molecular Structure Analyzer (Stable Version)")

# =========================================================
# 📂 UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload PDB file", type=["pdb"])

if uploaded_file:
    st.session_state.pdb_file = uploaded_file

    with open("temp.pdb", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDB uploaded!")

# =========================================================
# 🧪 PARSE STRUCTURE
# =========================================================
if st.session_state.pdb_file:

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", "temp.pdb")
        st.session_state.structure = structure

        coords, elements = [], []

        for atom in structure.get_atoms():
            coords.append(atom.get_coord())
            elements.append(atom.element)

        st.session_state.coords = np.array(coords)
        st.session_state.elements = elements

        st.success(f"Parsed successfully! Atoms: {len(coords)}")

    except Exception as e:
        st.error(f"Parsing error: {e}")

# =========================================================
# 🧬 3D VIEWER
# =========================================================
st.subheader("🧬 3D Viewer")

if st.session_state.pdb_file:
    with open("temp.pdb") as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=700, height=500)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"stick": {}})
    view.zoomTo()

    st.components.v1.html(view._make_html(), height=500)

# =========================================================
# 🔧 FIX STRUCTURE (SAFE)
# =========================================================
if st.button("Fix Structure"):

    if not PDBFIXER_AVAILABLE:
        st.error("❌ pdbfixer not installed. Skipping fixing step.")
    else:
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
if st.button("Build System"):

    try:
        from openmm.app import PDBFile
        pdb = PDBFile("temp.pdb")

        topology = (
            st.session_state.fixer.topology
            if st.session_state.fixer else pdb.topology
        )

        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        system = forcefield.createSystem(
            topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds
        )

        st.session_state.system = system
        st.success("System built!")

    except Exception as e:
        st.error(f"System error: {e}")

# =========================================================
# ▶️ SIMULATION
# =========================================================
if st.button("Run Simulation"):

    try:
        if st.session_state.system is None:
            st.warning("Build system first!")
        else:
            from openmm.app import PDBFile
            pdb = PDBFile("temp.pdb")

            integrator = LangevinIntegrator(
                300*kelvin,
                1/picosecond,
                0.002*picoseconds
            )

            if st.session_state.fixer:
                topology = st.session_state.fixer.topology
                positions = st.session_state.fixer.positions
            else:
                topology = pdb.topology
                positions = pdb.positions

            simulation = Simulation(
                topology,
                st.session_state.system,
                integrator
            )

            simulation.context.setPositions(positions)

            energies = []

            for i in range(50):
                simulation.step(10)
                state = simulation.context.getState(getEnergy=True)
                e = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
                energies.append(e)

            st.session_state.simulation = simulation
            st.session_state.energies = energies

            st.success("Simulation complete!")

    except Exception as e:
        st.error(f"Simulation error: {e}")

# =========================================================
# 📉 ENERGY PLOT
# =========================================================
if st.session_state.energies:

    st.subheader("📉 Energy vs Step")

    df = pd.DataFrame({
        "Step": range(len(st.session_state.energies)),
        "Energy": st.session_state.energies
    })

    st.line_chart(df.set_index("Step"))

# =========================================================
# 🧪 FORCE FIELD
# =========================================================
st.subheader("🧪 Force Field")

if st.session_state.system:

    system = st.session_state.system

    st.write(f"Particles: {system.getNumParticles()}")
    st.write(f"Forces: {system.getNumForces()}")

    for i in range(system.getNumForces()):
        force = system.getForce(i)
        st.write(f"{i}: {type(force).__name__}")

# =========================================================
# 📊 DSSP
# =========================================================
st.subheader("📊 DSSP Secondary Structure")

if st.session_state.structure:

    try:
        model = st.session_state.structure[0]
        dssp = DSSP(model, "temp.pdb")

        data = []
        for k in dssp.keys():
            res = dssp[k]
            data.append({
                "Residue": k[1][1],
                "AA": res[1],
                "Structure": res[2]
            })

        df = pd.DataFrame(data)
        st.dataframe(df)

    except:
        st.warning("DSSP not installed")

# =========================================================
# 📊 COORDINATES
# =========================================================
if st.session_state.coords is not None:

    st.subheader("📊 Coordinates")

    df = pd.DataFrame(st.session_state.coords, columns=["X", "Y", "Z"])
    st.dataframe(df.head())
    st.write(df.describe())
