# =============================
# FULL MOLECULAR MECHANICS APP
# =============================
import streamlit as st
import os
import copy
import numpy as np
import pandas as pd
from io import StringIO

# =============================
# OPTIONAL IMPORTS (SAFE)
# =============================
try:
    import mdtraj as md
    MDT_AVAILABLE = True
except:
    MDT_AVAILABLE = False

try:
    import parmed as pmd
    PARMED_AVAILABLE = True
except:
    PARMED_AVAILABLE = False

# =============================
# OPENMM
# =============================
from openmm.app import *
from openmm import *
from openmm.unit import *

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(layout="wide")
st.title("🧬 Full Molecular Mechanics Platform (CHARMM + vdW)")

# =============================
# OUTPUT FOLDER
# =============================
OUT = "charmm_dataset"
os.makedirs(OUT, exist_ok=True)

# =============================
# FILE UPLOAD
# =============================
uploaded_pdb = st.file_uploader("Upload Protein PDB", type=["pdb"])

# =============================
# VAN DER WAALS EXTRACTION
# =============================
def compute_vdw_energy(system, simulation):
    system_copy = copy.deepcopy(system)

    for force in system_copy.getForces():
        if isinstance(force, NonbondedForce):

            # Remove electrostatics → keep vdW only
            for i in range(force.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, 0.0, sigma, epsilon)

    # Create temporary simulation context
    integrator = VerletIntegrator(0.001)
    sim = Simulation(simulation.topology, system_copy, integrator)
    sim.context.setPositions(simulation.context.getState(getPositions=True).getPositions())

    state = sim.context.getState(getEnergy=True)
    vdw = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    return vdw

# =============================
# ENERGY COMPONENTS
# =============================
def get_energy_components(system, simulation):
    state = simulation.context.getState(getEnergy=True)
    total = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    energies = {"Total": total}

    for i, force in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        e = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        name = force.__class__.__name__

        if "HarmonicBondForce" in name:
            energies["Bond"] = e
        elif "HarmonicAngleForce" in name:
            energies["Angle"] = e
        elif "PeriodicTorsionForce" in name:
            energies["Dihedral"] = e
        elif "NonbondedForce" in name:
            energies["Nonbonded"] = e

    # 🔥 Split vdW and Electrostatic
    try:
        vdw = compute_vdw_energy(system, simulation)
        energies["vdW (LJ)"] = vdw
        energies["Electrostatic"] = energies["Nonbonded"] - vdw
    except:
        energies["vdW (LJ)"] = np.nan
        energies["Electrostatic"] = np.nan

    return energies

# =============================
# MAIN PIPELINE
# =============================
if uploaded_pdb:

    pdb_string = uploaded_pdb.read().decode("utf-8")
    pdb = PDBFile(StringIO(pdb_string))

    st.success("✅ PDB Loaded")

    # =============================
    # FORCE FIELD
    # =============================
    forcefield = ForceField(
        "charmm36.xml",
        "charmm36/water.xml"
    )

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*nanometer,
        constraints=HBonds
    )

    # Save system.xml
    with open(f"{OUT}/system.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))

    # =============================
    # OPTIONAL PSF EXPORT
    # =============================
    if PARMED_AVAILABLE:
        structure = pmd.openmm.load_topology(
            pdb.topology,
            system,
            pdb.positions
        )
        structure.save(f"{OUT}/topology.psf")
        st.success("✅ PSF generated")
    else:
        st.warning("⚠️ ParmEd not available → PSF skipped")

    # =============================
    # SIMULATION SETUP
    # =============================
    integrator = LangevinIntegrator(
        300*kelvin,
        1/picosecond,
        0.002*picoseconds
    )

    simulation = Simulation(
        pdb.topology,
        system,
        integrator
    )

    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    # =============================
    # OUTPUT FILES
    # =============================
    simulation.reporters.append(
        DCDReporter(f"{OUT}/trajectory.dcd", 10)
    )

    simulation.reporters.append(
        PDBReporter(f"{OUT}/output.pdb", 100)
    )

    # =============================
    # RUN CONTROL
    # =============================
    steps = st.slider("Simulation Steps", 100, 5000, 1000)

    if st.button("🚀 Run Simulation"):

        energies = []

        for step in range(steps):
            simulation.step(1)

            if step % 10 == 0:
                e = get_energy_components(system, simulation)
                e["Step"] = step
                energies.append(e)

        df = pd.DataFrame(energies)
        df.to_csv(f"{OUT}/energies.csv", index=False)

        st.success("✅ Simulation Completed")

        st.subheader("📊 Energy Data")
        st.dataframe(df.head(20))

        st.download_button(
            "Download Energies CSV",
            df.to_csv(index=False),
            file_name="energies.csv"
        )

# =============================
# THEORY SECTION
# =============================
st.markdown("## 📘 Energy Functions")

st.markdown("### Bond Energy")
st.latex("E_b = k_b (r - r_0)^2")

st.markdown("### Angle Energy")
st.latex("E_a = k_\\theta (\\theta - \\theta_0)^2")

st.markdown("### Dihedral Energy")
st.latex("E_d = k_d [1 + \\cos(n\\phi - \\delta)]")

st.markdown("### van der Waals (Lennard-Jones)")
st.latex("V(r) = 4\\epsilon [(\\sigma/r)^{12} - (\\sigma/r)^6]")

st.markdown("### Electrostatic Energy")
st.latex("E = \\frac{q_1 q_2}{4\\pi \\epsilon r}")

st.markdown("### Total Energy")
st.latex("""
E_{total} =
E_{bond} +
E_{angle} +
E_{dihedral} +
E_{vdW} +
E_{electrostatic}
""")

st.info("✅ Full CHARMM-style molecular mechanics with explicit vdW separation")
