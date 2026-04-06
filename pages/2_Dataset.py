import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

st.set_page_config(page_title="Molecular Mechanics Engine", layout="wide")

st.title("🧬 Molecular Mechanics Platform (CHARMM-style)")

# =========================================================
# ⚛️ ENERGY FUNCTIONS
# =========================================================

def bond_energy(coords, bonds, k_b, r0):
    E = 0
    for (i, j), kb, r_eq in zip(bonds, k_b, r0):
        r = np.linalg.norm(coords[i] - coords[j])
        E += kb * (r - r_eq)**2
    return E


def angle_energy(coords, angles, k_theta, theta0):
    E = 0
    for (i, j, k), kt, th0 in zip(angles, k_theta, theta0):
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]

        cos_theta = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        E += kt * (theta - th0)**2
    return E


def dihedral_energy(coords, dihedrals, k_phi, n, delta):
    E = 0
    for idx, (i, j, k, l) in enumerate(dihedrals):

        b1 = coords[j] - coords[i]
        b2 = coords[k] - coords[j]
        b3 = coords[l] - coords[k]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        phi = np.arccos(np.clip(np.dot(n1, n2), -1, 1))

        for t in range(len(k_phi[idx])):
            E += k_phi[idx][t] * (
                1 + np.cos(n[idx][t] * phi - delta[idx][t])
            )
    return E


# =========================================================
# 🌌 NONBONDED (vdW + Coulomb)
# =========================================================
def vdw_energy(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def coulomb_energy(q1, q2, r):
    k_e = 138.935456
    return k_e * q1 * q2 / r


def nonbonded_energy(coords, charges, sigma, epsilon,
                     exclusions=set(), one_four=set(), scale_14=0.5):

    E = 0
    N = len(coords)

    for i in range(N):
        for j in range(i+1, N):

            if (i, j) in exclusions:
                continue

            r = np.linalg.norm(coords[i] - coords[j]) + 1e-9

            sig = (sigma[i] + sigma[j]) / 2
            eps = np.sqrt(epsilon[i] * epsilon[j])

            vdw = vdw_energy(r, eps, sig)
            coul = coulomb_energy(charges[i], charges[j], r)

            if (i, j) in one_four:
                vdw *= scale_14
                coul *= scale_14

            E += vdw + coul

    return E


# =========================================================
# ⚛️ TOTAL ENERGY
# =========================================================
def total_energy(coords, bonds, bond_k, bond_r0,
                 angles, angle_k, angle_theta0,
                 dihedrals, dih_k, dih_n, dih_delta,
                 charges, sigma, epsilon,
                 exclusions, one_four):

    Eb = bond_energy(coords, bonds, bond_k, bond_r0)
    Ea = angle_energy(coords, angles, angle_k, angle_theta0)
    Ed = dihedral_energy(coords, dihedrals, dih_k, dih_n, dih_delta)
    Enb = nonbonded_energy(coords, charges, sigma, epsilon,
                           exclusions, one_four)

    return Eb, Ea, Ed, Enb, Eb + Ea + Ed + Enb


# =========================================================
# 📂 FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("Upload PDB or CSV (CHARMM/AMBER style)")

if uploaded_file:

    text = uploaded_file.read().decode("utf-8")
    lines = text.splitlines()

    # =====================================================
    # 🧬 PDB PARSER
    # =====================================================
    if any(line.startswith(("ATOM", "HETATM")) for line in lines):
        st.success("🧬 PDB Detected")

        atoms, coords = [], []

        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                atoms.append(parts[2])
                coords.append([float(parts[6]), float(parts[7]), float(parts[8])])

        coords = np.array(coords)
        N = len(coords)

        st.write("Atoms:", atoms)

        # =================================================
        # 🔗 AUTO TOPOLOGY
        # =================================================
        bonds = [(i, i+1) for i in range(N-1)]
        angles = [(i, i+1, i+2) for i in range(N-2)]
        dihedrals = [(i, i+1, i+2, i+3) for i in range(N-3)]

        exclusions = set(bonds + [(i, i+2) for i in range(N-2)])
        one_four = set([(i, i+3) for i in range(N-3)])

        # =================================================
        # ⚙️ PARAMETERS (PLACEHOLDER → replace with CHARMM)
        # =================================================
        bond_k = [300]*len(bonds)
        bond_r0 = [1.5]*len(bonds)

        angle_k = [40]*len(angles)
        angle_theta0 = [np.pi/2]*len(angles)

        dih_k = [[2, 1]]*len(dihedrals)
        dih_n = [[3, 2]]*len(dihedrals)
        dih_delta = [[0, np.pi/2]]*len(dihedrals)

        charges = np.random.uniform(-0.5, 0.5, N)
        sigma = np.random.uniform(1.0, 2.0, N)
        epsilon = np.random.uniform(0.1, 0.5, N)

        # =================================================
        # ⚛️ ENERGY COMPUTATION
        # =================================================
        Eb, Ea, Ed, Enb, Etot = total_energy(
            coords,
            bonds, bond_k, bond_r0,
            angles, angle_k, angle_theta0,
            dihedrals, dih_k, dih_n, dih_delta,
            charges, sigma, epsilon,
            exclusions, one_four
        )

        st.subheader("⚛️ Energy Components")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bond", f"{Eb:.3f}")
        c2.metric("Angle", f"{Ea:.3f}")
        c3.metric("Dihedral", f"{Ed:.3f}")
        c4.metric("vdW + Coulomb", f"{Enb:.3f}")
        c5.metric("Total", f"{Etot:.3f}")

    # =====================================================
    # 📊 CSV PARSER (CHARMM-like)
    # =====================================================
    else:
        st.success("📊 CSV Detected")

        df = pd.read_csv(StringIO(text))
        st.dataframe(df)

        required = {"charge", "sigma", "epsilon"}

        if not required.issubset(df.columns):
            st.warning("⚠️ Missing FF parameters → auto-generating")
            df["charge"] = df.get("charge", 0)
            df["sigma"] = np.random.uniform(1, 2, len(df))
            df["epsilon"] = np.random.uniform(0.1, 0.5, len(df))

        st.subheader("⚛️ Pair Interaction")

        i = st.selectbox("Particle i", df.index)
        j = st.selectbox("Particle j", df.index, index=1)

        r = st.slider("Distance", 0.5, 10.0, 2.0)

        sig = (df.loc[i, "sigma"] + df.loc[j, "sigma"]) / 2
        eps = np.sqrt(df.loc[i, "epsilon"] * df.loc[j, "epsilon"])

        vdw = vdw_energy(r, eps, sig)
        coul = coulomb_energy(df.loc[i, "charge"], df.loc[j, "charge"], r)

        st.metric("vdW Energy", f"{vdw:.4f}")
        st.metric("Coulomb Energy", f"{coul:.4f}")
        st.metric("Total", f"{vdw + coul:.4f}")

else:
    st.info("Upload a dataset to begin")
