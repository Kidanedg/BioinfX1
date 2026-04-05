st.title("⚙️ Simulation Engine")

show_details = st.checkbox("Show Detailed Energy Breakdown")

if show_details:
    st.write(f"Bond Energy: {bond_energy:.2f}")
    st.write(f"Angle Energy: {angle_energy:.2f}")
    st.write(f"vdW Energy: {vdw_energy:.2f}")
    st.write(f"Electrostatic Energy: {elec_energy:.2f}")
