# =============================
# FULL MOLECULAR MECHANICS APP
# ERROR-PROOF VERSION
# =============================
import streamlit as st
import os
import copy
import numpy as np
import pandas as pd
from io import StringIO

# =============================
# SAFE IMPORTS
# =============================
try:
    import parmed as pmd
    PARMED_AVAILABLE = True
except:
    PARMED_AVAILABLE = False

# OpenMM
from openmm.app import *
from openmm import *
from openmm.unit import *

# =============================
# UI
# =============================
st.set_page_config(layout="wide")
st.title("🧬 Robust Molecular Mechanics Platform (CHARMM + vdW)")

OUT = "charmm_dataset"
os.makedirs(OUT, exist_ok=True)

uploaded_pdb = st.file_uploader("Upload PDB File", type=["pdb"])

# =============================
# SAFE PDB LOADER
# =============================
def safe_load_pdb(uploaded_file):
    try:
        pdb_string = uploaded_file.read().decode("utf-8")
        pdb = PDBFile(StringIO(pdb_string))
        return pdb
    except Exception as e:
        st.error(f"❌ PDB load failed: {e}")
        return None

# =============================
# STRUCTURE FIXER
# =============================
def prepare_structure(pdb, forcefield):
    try:
        modeller = Modeller(pdb.topology, pdb.positions)

        # Add hydrogens (fix ALA/GLY errors)
        modeller.addHydrogens(forcefield)

        return modeller
    except Exception as e:
        st.warning(f"⚠️ Hydrogen addition failed: {e}")
        return pdb

# =============================
# SAFE SYSTEM CREATION
# =============================
def create_safe_system(topology, forcefield):
    try:
        if topology.getPeriodicBoxVectors() is None:
            st.warning("⚠️ No box → Using NoCutoff (vacuum)")
            system = forcefield.createSystem(
                topology,
                nonbondedMethod=NoCutoff,
                constraints=HBonds
            )
        else:
            system = forcefield.createSystem(
                topology,
                nonbondedMethod=PME,
                nonbondedCutoff=1.0*nanometer,
                constraints=HBonds
            )
        return system

    except Exception as e:
        st.error(f"❌ System creation failed: {e}")
        return None

# =============================
# vdW ENERGY
# =============================
def compute_vdw_energy(system, simulation):
    try:
        system_copy = copy.deepcopy(system)

        for force in system_copy.getForces():
            if isinstance(force, NonbondedForce):
                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    force.setParticleParameters(i, 0.0, sigma, epsilon)

        integrator = VerletIntegrator(0.001)
        sim = Simulation(simulation.topology, system_copy, integrator)

        sim.context.setPositions(
            simulation.context.getState(getPositions=True).getPositions()
        )

        state = sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    except Exception:
        return np.nan

# =============================
# ENERGY COMPONENTS
# =============================
def get_energy(system, simulation):
    try:
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

        # vdW split
        vdw = compute_vdw_energy(system, simulation)
        energies["vdW"] = vdw
        energies["Electrostatic"] = energies.get("Nonbonded", 0) - vdw

        return energies

    except Exception as e:
        return {"Error": str(e)}

# =============================
# MAIN
# =============================
if uploaded_pdb:

    pdb = safe_load_pdb(uploaded_pdb)

    if pdb is not None:
        st.success("✅ PDB Loaded")

        # =============================
        # FORCE FIELD (SAFE LOAD)
        # =============================
        try:
            forcefield = ForceField("charmm36.xml", "charmm36/water.xml")
        except:
            st.error("❌ CHARMM force field missing in environment")
            st.stop()

        # =============================
        # STRUCTURE FIX
        # =============================
        modeller = prepare_structure(pdb, forcefield)

        # =============================
        # SYSTEM
        # =============================
        system = create_safe_system(modeller.topology, forcefield)

        if system is None:
            st.stop()

        # Save system
        try:
            with open(f"{OUT}/system.xml", "w") as f:
                f.write(XmlSerializer.serialize(system))
        except:
            st.warning("⚠️ Could not save system.xml")

        # =============================
        # OPTIONAL PSF
        # =============================
        if PARMED_AVAILABLE:
            try:
                structure = pmd.openmm.load_topology(
                    modeller.topology,
                    system,
                    modeller.positions
                )
                structure.save(f"{OUT}/topology.psf")
                st.success("✅ PSF saved")
            except:
                st.warning("⚠️ PSF generation failed")

        # =============================
        # SIMULATION
        # =============================
        integrator = LangevinIntegrator(
            300*kelvin,
            1/picosecond,
            0.002*picoseconds
        )

        simulation = Simulation(
            modeller.topology,
            system,
            integrator
        )

        try:
            simulation.context.setPositions(modeller.positions)
            simulation.minimizeEnergy()
        except Exception as e:
            st.error(f"❌ Simulation init failed: {e}")
            st.stop()

        # =============================
        # REPORTERS
        # =============================
        try:
            simulation.reporters.append(
                DCDReporter(f"{OUT}/traj.dcd", 10)
            )
        except:
            st.warning("⚠️ DCD disabled")

        # =============================
        # RUN
        # =============================
        steps = st.slider("Steps", 100, 3000, 1000)

        if st.button("🚀 Run Simulation"):

            energies = []

            for step in range(steps):
                simulation.step(1)

                if step % 10 == 0:
                    e = get_energy(system, simulation)
                    e["Step"] = step
                    energies.append(e)

            df = pd.DataFrame(energies)

            st.success("✅ Simulation Completed")
            st.dataframe(df.head())

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "energies.csv"
            )

# =============================
# THEORY
# =============================
st.markdown("## 📘 Energy Model")

st.latex("E_{total} = E_b + E_a + E_d + E_{vdW} + E_{elec}")
st.latex("E_b = k_b (r - r_0)^2")
st.latex("E_a = k_\\theta (\\theta - \\theta_0)^2")
st.latex("E_d = k_d [1 + \\cos(n\\phi - \\delta)]")
st.latex("V_{vdW} = 4\\epsilon [(\\sigma/r)^{12} - (\\sigma/r)^6]")
st.latex("E_{elec} = \\frac{q_1 q_2}{4\\pi \\epsilon r}")

st.info("✅ Fully robust CHARMM-style molecular mechanics engine")
