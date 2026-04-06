import streamlit as st

st.title("📘 Force Field Theory")

# =============================
# 📌 INTRO
# =============================
st.markdown("""
Molecular mechanics describes a system’s energy using classical physics.
The **total potential energy** is the sum of bonded and non-bonded interactions.
""")

# =============================
# 🔗 BOND ENERGY
# =============================
st.markdown("## 🔗 Bond Energy")

st.markdown("""
Bond stretching is modeled as a harmonic spring.
Atoms prefer an equilibrium distance, and deviations cost energy.
""")

:contentReference[oaicite:0]{index=0}

st.markdown("""
- **k** → force constant (bond stiffness)  
- **r₀** → equilibrium bond length  
- **r** → current bond length  

👉 If the bond stretches or compresses, energy increases.
""")

# =============================
# 🌌 LENNARD-JONES
# =============================
st.markdown("## 🌌 Lennard-Jones Potential")

st.markdown("""
This describes **van der Waals interactions**:
- Attraction at long distance  
- Repulsion at short distance  
""")

:contentReference[oaicite:1]{index=1}

st.markdown("""
- **ε (epsilon)** → interaction strength  
- **σ (sigma)** → distance where potential = 0  

👉 The **r⁻¹² term** = strong repulsion  
👉 The **r⁻⁶ term** = weak attraction  
""")

# =============================
# ⚡ TOTAL ENERGY FUNCTION
# =============================
st.markdown("## ⚡ Total Potential Energy")

st.markdown("""
The total energy of a molecular system is the sum of all interactions:
""")

:contentReference[oaicite:2]{index=2}

st.markdown("""
### 🔍 Breakdown:

**1. Bond Stretching**
- Energy from bond length changes

**2. Angle Bending**
- Energy from bond angle deviations

**3. Dihedral (Torsion)**
- Rotation around bonds

**4. Non-bonded Interactions**
- Lennard-Jones (van der Waals)
- (Often includes electrostatics too)

👉 This equation is the **foundation of molecular dynamics simulations (OpenMM, AMBER, CHARMM)**.
""")

# =============================
# 🧠 LEARNING NOTE
# =============================
st.info("""
🎓 Students must understand how each term contributes to total energy:
- Bonded terms stabilize structure
- Non-bonded terms control interactions and folding
""")
