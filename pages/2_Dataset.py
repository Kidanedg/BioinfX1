import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO

# =============================
# OPENMM IMPORTS (OPTIONAL)
# =============================
try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    OPENMM_AVAILABLE = True
except:
    OPENMM_AVAILABLE = False

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Full CHARMM Engine", layout="wide")
st.title("🧬 Full Molecular Mechanics Engine (CHARMM-style)")

# =============================
# LOAD PDB
# =============================
def load_structure(file):
    pdb_string = file.read().decode("utf-8")

    lines = [
        l for l in pdb_string.splitlines()
        if l.startswith(("ATOM", "HETATM"))
    ]

    if not lines:
        st.error("❌ No valid atoms found in PDB")
        return None

    fixed = "\n".join(lines) + "\nEND\n"
    return PDBFile(StringIO(fixed))


# =============================
# EXTRACT COORDS
# =============================
def extract_data(pdb):
    coords = np.array([
        [p.x, p.y, p.z]
        for p in pdb.positions
    ])
    elements = [
        atom.element.symbol if atom.element else "C"
        for atom in pdb.topology.atoms()
    ]
    return coords, elements


# =============================
# PARAMETERS
# =============================
def get_params(elements):
    kb, r0 = 300, 0.14
    ka, theta0 = 40, np.deg2rad(109.5)
    kd, n, delta = 2, 3, 0

    charge_map = {"H":0.1,"C":-0.1,"O":-0.5,"N":-0.3}
    charges = np.array([charge_map.get(e,0) for e in elements])

    return kb, r0, ka, theta0, kd, n, delta, charges


# =============================
# TOPOLOGY DETECTION
# =============================
def detect_bonds(coords, cutoff=0.2):
    bonds = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            if np.linalg.norm(coords[i]-coords[j]) < cutoff:
                bonds.append((i,j))
    return bonds


def detect_angles(bonds):
    angles = []
    for i,j in bonds:
        for k,l in bonds:
            if j == k and i != l:
                angles.append((i,j,l))
    return angles


def detect_dihedrals(bonds):
    dihedrals = []
    for i,j in bonds:
        for k,l in bonds:
            if j == k:
                for m,n in bonds:
                    if l == m:
                        dihedrals.append((i,j,k,n))
    return dihedrals


# =============================
# ENERGY FUNCTIONS
# =============================

# Bond: harmonic
def bond_energy(coords,bonds,kb,r0):
    return sum(
        kb*(np.linalg.norm(coords[i]-coords[j]) - r0)**2
        for i,j in bonds
    )

# Angle: harmonic
def angle_energy(coords,angles,ka,theta0):
    E = 0
    for i,j,k in angles:
        v1 = coords[i]-coords[j]
        v2 = coords[k]-coords[j]

        cos_theta = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta,-1,1))

        E += ka*(theta-theta0)**2
    return E

# Dihedral: periodic
def dihedral_energy(coords,dihedrals,kd,n,delta):
    E = 0
    for i,j,k,l in dihedrals:
        b1 = coords[j]-coords[i]
        b2 = coords[k]-coords[j]
        b3 = coords[l]-coords[k]

        n1 = np.cross(b1,b2)
        n2 = np.cross(b2,b3)

        phi = np.arccos(
            np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2))
        )

        E += kd*(1 + np.cos(n*phi - delta))
    return E

# =============================
# NONBONDED
# =============================

# Lennard-Jones (van der Waals)
def vdw_energy(coords):
    eps, sigma = 0.2, 0.34
    E = 0

    for i in range(len(coords)):
        for j in range(i+1,len(coords)):
            r = np.linalg.norm(coords[i]-coords[j])
            if r < 1e-6:
                continue

            E += 4*eps*((sigma/r)**12 - (sigma/r)**6)

    return E

# Coulomb
def electro_energy(coords,charges):
    k = 138.9
    E = 0

    for i in range(len(coords)):
        for j in range(i+1,len(coords)):
            r = np.linalg.norm(coords[i]-coords[j])
            if r < 1e-6:
                continue

            E += k*charges[i]*charges[j]/r

    return E


# =============================
# TOTAL ENERGY FUNCTION
# =============================
def total_energy(coords,elements):
    kb,r0,ka,theta0,kd,n,delta,charges = get_params(elements)

    bonds = detect_bonds(coords)
    angles = detect_angles(bonds)
    dihedrals = detect_dihedrals(bonds)

    Eb = bond_energy(coords,bonds,kb,r0)
    Ea = angle_energy(coords,angles,ka,theta0)
    Ed = dihedral_energy(coords,dihedrals,kd,n,delta)
    Ev = vdw_energy(coords)
    Ee = electro_energy(coords,charges)

    return {
        "Bond": Eb,
        "Angle": Ea,
        "Dihedral": Ed,
        "VdW (LJ)": Ev,
        "Electrostatic": Ee,
        "TOTAL": Eb + Ea + Ed + Ev + Ee
    }


# =============================
# UI
# =============================
pdb_file = st.file_uploader("Upload PDB File")

if pdb_file:
    pdb = load_structure(pdb_file)

    if pdb:
        coords, elements = extract_data(pdb)

        st.success(f"✅ Loaded {len(coords)} atoms")

        if st.button("Compute Full Energy"):
            energies = total_energy(coords,elements)
            st.dataframe(pd.DataFrame([energies]))

        # Optional OpenMM
        if OPENMM_AVAILABLE and st.button("Run OpenMM Minimization"):
            system = ForceField('amber14-all.xml','amber14/tip3pfb.xml').createSystem(
                pdb.topology
            )

            integrator = LangevinIntegrator(
                300*kelvin, 1/picosecond, 0.002*picoseconds
            )

            simulation = Simulation(
                pdb.topology, system, integrator
            )

            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy()

            state = simulation.context.getState(getEnergy=True)
            st.write("⚡ OpenMM Energy:", state.getPotentialEnergy())
