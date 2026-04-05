import streamlit as st

st.title("📘 Force Field Theory")

st.markdown("""
### Bond Energy

Bond energy describes the energy required to stretch or compress a bond:
E = k(r - r0)^2
""")

st.markdown("""
### Lennard-Jones Potential

The Lennard-Jones potential models van der Waals interactions:
V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
""")

st.info("Students must understand equations before simulation.")
