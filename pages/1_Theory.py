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

st.latex(r"E = k(r - r_0)^2")

st.markdown("""
- **k** → force constant (bond stiffness)  
- **r₀** → equilibrium bond length  
- **r** → current bond length  
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

st.latex(r"V(r) = 4\varepsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]")

st.markdown("""
- **ε (epsilon)** → interaction strength  
- **σ (sigma)** → distance where potential = 0  
""")

# =============================
# ⚡ TOTAL ENERGY FUNCTION
# =============================
st.markdown("## ⚡ Total Potential Energy")

st.latex(r"""
E_{total} =
\sum_{bonds} k_b(r - r_0)^2 +
\sum_{angles} k_\theta(\theta - \theta_0)^2 +
\sum_{dihedrals} V_n[1 + \cos(n\phi - \gamma)] +
\sum_{i<j} 4\varepsilon \left[
\left(\frac{\sigma}{r_{ij}}\right)^{12} -
\left(\frac{\sigma}{r_{ij}}\right)^6
\right]
""")

st.markdown("""
### 🔍 Breakdown:

**1. Bond Stretching** → bond length changes  
**2. Angle Bending** → bond angle deviations  
**3. Dihedral (Torsion)** → rotation around bonds  
**4. Non-bonded** → van der Waals interactions  

👉 This is the core equation behind **OpenMM simulations**.
""")

# =============================
# 🧠 NOTE
# =============================
st.info("🎓 Students must understand equations before simulation.")
