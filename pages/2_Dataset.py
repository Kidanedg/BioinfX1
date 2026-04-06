# =============================
# 🧬 FULL MOLECULAR DATASET + DOCKING ENGINE
# =============================
import streamlit as st
import os, copy
import numpy as np
import pandas as pd
from io import StringIO

from openmm.app import *
from openmm import *
from openmm.unit import *

# =============================
# UI CONFIG
# =============================
st.set_page_config(layout="wide")
st.title("🧬 Molecular Dataset Generator + Docking Scoring")

OUT = "dataset_output"
os.makedirs(OUT, exist_ok=True)

uploaded_pdb = st.file_uploader("📂 Upload PDB File", type=["pdb"])

# =============================
# 🔁 PIPELINE FLOWCHART
# =============================
st.markdown("## 🔄 Pipeline")

st.code("""
Upload PDB
   ↓
Detect type (protein / ligand)
   ↓
Load CHARMM or AMBER
   ↓
Fix structure (hydrogens)
   ↓
Create system (PME or vacuum)
   ↓
Run simulation
   ↓
Extract energies (Eb, Ea, Ed, vdW, Eelec)
   ↓
Add geometry features
   ↓
Export dataset.csv
""")

# =============================
# DETECT TYPE
# =============================
def detect_type(topology):
    protein_res = {
        "ALA","GLY","VAL","LEU","ILE","SER","THR","ASP","GLU",
        "ASN","GLN","LYS","ARG","HIS","PHE","TYR","TRP","PRO"
    }
    names = set([r.name for r in topology.residues()])
    return "protein" if names.issubset(protein_res) else "ligand"

# =============================
# LOAD PDB
# =============================
def load_pdb(file):
    try:
        return PDBFile(StringIO(file.read().decode("utf-8")))
    except Exception as e:
        st.error(f"PDB error: {e}")
        return None

# =============================
# LOAD FORCE FIELD
# =============================
def load_ff(mol_type):
    try:
        if mol_type == "protein":
            return ForceField("charmm36.xml", "charmm36/water.xml")
        else:
            return ForceField("amber14-all.xml")
    except Exception as e:
        st.error(f"ForceField error: {e}")
        return None

# =============================
# PREPARE STRUCTURE
# =============================
def prepare(pdb, ff, mol_type):
    modeller = Modeller(pdb.topology, pdb.positions)
    try:
        modeller.addHydrogens(ff)
    except:
        pass
    return modeller

# =============================
# CREATE SYSTEM
# =============================
def create_system(topology, ff):
    try:
        if topology.getPeriodicBoxVectors() is None:
            st.warning("⚠️ Vacuum mode (NoCutoff)")
            return ff.createSystem(
                topology,
                nonbondedMethod=NoCutoff,
                constraints=None
            )
        else:
            return ff.createSystem(
                topology,
                nonbondedMethod=PME,
                nonbondedCutoff=1.0*nanometer
            )
    except Exception as e:
        st.error(e)
        return None

# =============================
# vdW ENERGY
# =============================
def compute_vdw(system, simulation):
    try:
        sys_copy = copy.deepcopy(system)

        for f in sys_copy.getForces():
            if isinstance(f, NonbondedForce):
                for i in range(f.getNumParticles()):
                    q,s,e = f.getParticleParameters(i)
                    f.setParticleParameters(i, 0.0, s, e)

        integ = VerletIntegrator(0.001)
        sim = Simulation(simulation.topology, sys_copy, integ)

        sim.context.setPositions(
            simulation.context.getState(getPositions=True).getPositions()
        )

        state = sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    except:
        return np.nan

# =============================
# DISTANCE FEATURE
# =============================
def distance(sim):
    try:
        pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
        return np.linalg.norm(pos[0] - pos[1])
    except:
        return np.nan

# =============================
# ENERGY EXTRACTION
# =============================
def get_energy(system, sim):
    try:
        state = sim.context.getState(getEnergy=True)
        total = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        E = {"Total": total}

        for i, f in enumerate(system.getForces()):
            s = sim.context.getState(getEnergy=True, groups={i})
            val = s.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

            name = f.__class__.__name__

            if "Bond" in name:
                E["Bond"] = val
            elif "Angle" in name:
                E["Angle"] = val
            elif "Torsion" in name:
                E["Dihedral"] = val
            elif "Nonbonded" in name:
                E["Nonbonded"] = val

        vdw = compute_vdw(system, sim)
        E["vdW"] = vdw
        E["Electrostatic"] = E.get("Nonbonded", 0) - vdw

        return E

    except Exception as e:
        return {"error": str(e)}

# =============================
# DATASET GENERATOR
# =============================
def generate(sim, system, steps, interval):
    data = []

    for step in range(steps):
        sim.step(1)

        if step % interval == 0:
            e = get_energy(system, sim)

            data.append({
                "Step": step,
                "Bond": e.get("Bond",0),
                "Angle": e.get("Angle",0),
                "Dihedral": e.get("Dihedral",0),
                "vdW": e.get("vdW",0),
                "Electrostatic": e.get("Electrostatic",0),
                "Total": e.get("Total",0),
                "Distance_0_1": distance(sim)
            })

    return pd.DataFrame(data)

# =============================
# DOCKING SCORE (ΔE)
# =============================
def docking_score(df):
    try:
        # simple scoring: lower total energy = better binding
        return df["Total"].min()
    except:
        return np.nan

# =============================
# MAIN PIPELINE
# =============================
if uploaded_pdb:

    pdb = load_pdb(uploaded_pdb)

    if pdb:
        mol_type = detect_type(pdb.topology)
        st.success(f"✅ Detected: {mol_type}")

        ff = load_ff(mol_type)
        if ff is None:
            st.stop()

        modeller = prepare(pdb, ff, mol_type)

        system = create_system(modeller.topology, ff)
        if system is None:
            st.stop()

        integrator = LangevinIntegrator(
            300*kelvin,
            1/picosecond,
            0.002*picoseconds
        )

        sim = Simulation(modeller.topology, system, integrator)

        sim.context.setPositions(modeller.positions)
        sim.minimizeEnergy()

        # UI controls
        steps = st.slider("Simulation Steps", 100, 5000, 1000)
        interval = st.slider("Sampling Interval", 1, 50, 10)

        if st.button("🚀 Run Simulation & Generate Dataset"):

            df = generate(sim, system, steps, interval)

            score = docking_score(df)

            st.success("✅ Dataset Generated")
            st.dataframe(df.head())

            # Show docking score
            st.metric("🧲 Docking Score (Lower = Better)", f"{score:.3f} kJ/mol")

            # Save
            file_path = f"{OUT}/dataset.csv"
            df.to_csv(file_path, index=False)

            st.download_button(
                "⬇️ Download dataset.csv",
                df.to_csv(index=False),
                "dataset.csv"
            )

# =============================
# THEORY SECTION
# =============================
st.markdown("## 📘 Energy Model")

st.latex("E_{total}=E_b+E_a+E_d+E_{vdW}+E_{elec}")
st.latex("E_b=k_b(r-r_0)^2")
st.latex("E_a=k_\\theta(\\theta-\\theta_0)^2")
st.latex("E_d=k_d[1+\\cos(n\\phi-\\delta)]")
st.latex("V_{vdW}=4\\epsilon[(\\sigma/r)^{12}-(\\sigma/r)^6]")
st.latex("E_{elec}=\\frac{q_1q_2}{4\\pi\\epsilon r}")

st.info("✅ Supports protein + ligand + vdW + docking scoring + dataset export")
