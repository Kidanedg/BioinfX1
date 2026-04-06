# =============================
# 🧬 FULL REAL PDB DATASET PIPELINE (FINAL CLEAN VERSION)
# =============================
import os
import requests
import copy
import numpy as np
import pandas as pd

from openmm.app import *
from openmm import *
from openmm.unit import *

# =============================
# CONFIG
# =============================
OUT_DIR = "pdb_dataset"
DATASET_FILE = "final_pdb_dataset.csv"

# =============================
# STEP 1 — DOWNLOAD PDB FILES
# =============================
def download_pdb(pdb_ids, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    for pid in pdb_ids:
        url = f"https://files.rcsb.org/download/{pid}.pdb"
        try:
            r = requests.get(url, timeout=15)

            if r.status_code == 200:
                with open(f"{out_dir}/{pid}.pdb", "w") as f:
                    f.write(r.text)
                print(f"✅ Downloaded {pid}")
            else:
                print(f"❌ Failed {pid}")

        except Exception as e:
            print(f"⚠️ Error {pid}: {e}")

# =============================
# STEP 2 — DETECT TYPE
# =============================
def detect_type(topology):
    protein_res = {
        "ALA","GLY","VAL","LEU","ILE","SER","THR","ASP","GLU",
        "ASN","GLN","LYS","ARG","HIS","PHE","TYR","TRP","PRO"
    }

    names = set([r.name for r in topology.residues()])

    # if majority are protein residues → protein
    protein_count = sum([1 for r in names if r in protein_res])

    return "protein" if protein_count > len(names)/2 else "ligand"

# =============================
# STEP 3 — LOAD FORCE FIELD
# =============================
def load_ff(mol_type):
    try:
        if mol_type == "protein":
            return ForceField("charmm36.xml", "charmm36/water.xml")
        else:
            return ForceField("amber14-all.xml")
    except Exception as e:
        print(f"⚠️ FF load failed ({mol_type}), fallback to AMBER: {e}")
        try:
            return ForceField("amber14-all.xml")
        except:
            return None

# =============================
# STEP 4 — PREPARE STRUCTURE
# =============================
def prepare(pdb, ff):
    modeller = Modeller(pdb.topology, pdb.positions)

    try:
        modeller.addHydrogens(ff)
    except Exception as e:
        print("⚠️ Hydrogen addition skipped:", e)

    return modeller

# =============================
# STEP 5 — CREATE SYSTEM
# =============================
def create_system(topology, ff):
    try:
        return ff.createSystem(
            topology,
            nonbondedMethod=NoCutoff
        )
    except Exception as e:
        print("⚠️ System creation failed:", e)
        return None

# =============================
# STEP 6 — vdW ENERGY (FIXED)
# =============================
def compute_vdw(system, simulation):
    try:
        sys_copy = copy.deepcopy(system)

        for f in sys_copy.getForces():
            if isinstance(f, NonbondedForce):
                for i in range(f.getNumParticles()):
                    q, s, e = f.getParticleParameters(i)
                    # remove electrostatics → keep vdW
                    f.setParticleParameters(i, 0.0, s, e)

        integrator = VerletIntegrator(0.001)
        sim = Simulation(simulation.topology, sys_copy, integrator)

        sim.context.setPositions(
            simulation.context.getState(getPositions=True).getPositions()
        )

        state = sim.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    except Exception as e:
        print("⚠️ vdW error:", e)
        return np.nan

# =============================
# STEP 7 — ENERGY EXTRACTION
# =============================
def get_energy(system, sim):
    try:
        state = sim.context.getState(getEnergy=True)
        total = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        E = {"Total": total}

        for i, f in enumerate(system.getForces()):
            sim.context.reinitialize(preserveState=True)

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
        print("⚠️ Energy error:", e)
        return {}

# =============================
# STEP 8 — SIMULATION
# =============================
def simulate(modeller, system):
    integrator = LangevinIntegrator(
        300*kelvin,
        1/picosecond,
        0.002*picoseconds
    )

    sim = Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)

    try:
        sim.minimizeEnergy()
    except Exception as e:
        print("⚠️ Minimization failed:", e)

    return sim

# =============================
# STEP 9 — PROCESS SINGLE PDB
# =============================
def process_pdb(pdb_path):
    try:
        pdb = PDBFile(pdb_path)

        mol_type = detect_type(pdb.topology)
        ff = load_ff(mol_type)

        if ff is None:
            print(f"❌ FF failed for {pdb_path}")
            return None

        modeller = prepare(pdb, ff)
        system = create_system(modeller.topology, ff)

        if system is None:
            return None

        sim = simulate(modeller, system)

        e = get_energy(system, sim)

        return {
            "PDB_ID": os.path.basename(pdb_path).replace(".pdb",""),
            "Type": mol_type,
            "Bond": e.get("Bond", np.nan),
            "Angle": e.get("Angle", np.nan),
            "Dihedral": e.get("Dihedral", np.nan),
            "vdW": e.get("vdW", np.nan),
            "Electrostatic": e.get("Electrostatic", np.nan),
            "Total": e.get("Total", np.nan),
            "DockingScore": e.get("Total", np.nan)
        }

    except Exception as e:
        print(f"❌ Error {pdb_path}: {e}")
        return None

# =============================
# STEP 10 — BUILD DATASET
# =============================
def build_dataset(folder=OUT_DIR):
    data = []

    for file in os.listdir(folder):
        if file.endswith(".pdb"):
            print(f"🔬 Processing {file}...")

            row = process_pdb(os.path.join(folder, file))

            if row:
                data.append(row)

    if len(data) == 0:
        print("❌ No valid data generated")
        return None

    df = pd.DataFrame(data)
    df.to_csv(DATASET_FILE, index=False)

    print("\n✅ DATASET SAVED:", DATASET_FILE)
    return df

# =============================
# RUN PIPELINE
# =============================
if __name__ == "__main__":

    pdb_ids = ["1CRN", "4HHB", "1A2K", "2PTC", "3CLN"]

    print("⬇️ Downloading PDBs...")
    download_pdb(pdb_ids)

    print("\n⚙️ Building dataset...")
    df = build_dataset()

    if df is not None:
        print("\n📊 Preview:")
        print(df.head())
