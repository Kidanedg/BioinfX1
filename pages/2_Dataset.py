import os
import numpy as np
import pandas as pd

from openmm.app import *
from openmm import *
from openmm.unit import *

import mdtraj as md
import parmed as pmd

# =============================
# OUTPUT FOLDER
# =============================
OUT = "charmm_dataset"
os.makedirs(OUT, exist_ok=True)

# =============================
# LOAD PDB (INPUT STRUCTURE)
# =============================
pdb = PDBFile("input.pdb")   # <-- replace with your real protein

# =============================
# CHARMM FORCE FIELD
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

# =============================
# SAVE SYSTEM (PRM EQUIVALENT)
# =============================
with open(f"{OUT}/system.xml", "w") as f:
    f.write(XmlSerializer.serialize(system))

# =============================
# CONVERT TO PSF (TRUE CHARMM)
# =============================
structure = pmd.openmm.load_topology(
    pdb.topology,
    system,
    pdb.positions
)

structure.save(f"{OUT}/topology.psf")

# =============================
# SIMULATION SETUP
# =============================
integrator = LangevinIntegrator(
    300*kelvin,
    1/picosecond,
    0.002*picoseconds
)

platform = Platform.getPlatformByName("CPU")

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform
)

simulation.context.setPositions(pdb.positions)

# =============================
# MINIMIZATION
# =============================
simulation.minimizeEnergy()

# =============================
# OUTPUT FILES
# =============================
simulation.reporters.append(
    DCDReporter(f"{OUT}/trajectory.dcd", 10)
)

simulation.reporters.append(
    PDBReporter(f"{OUT}/structure.pdb", 100)
)

# =============================
# ENERGY TRACKING
# =============================
energies = []

class EnergyReporter(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, simulation, state):
        pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        ke = state.getKineticEnergy().value_in_unit(kilojoule_per_mole)

        energies.append({
            "step": simulation.currentStep,
            "Potential": pe,
            "Kinetic": ke,
            "Total": pe + ke
        })

simulation.reporters.append(
    StateDataReporter(
        stdout,
        10,
        step=True,
        potentialEnergy=True,
        temperature=True
    )
)

# =============================
# RUN SIMULATION
# =============================
n_steps = 1000

for step in range(n_steps):
    simulation.step(1)

    if step % 10 == 0:
        state = simulation.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        ke = state.getKineticEnergy().value_in_unit(kilojoule_per_mole)

        energies.append({
            "step": step,
            "Potential": pe,
            "Kinetic": ke,
            "Total": pe + ke
        })

# =============================
# SAVE ENERGY DATA
# =============================
df = pd.DataFrame(energies)
df.to_csv(f"{OUT}/energies.csv", index=False)

print("✅ REAL CHARMM dataset generated!")
