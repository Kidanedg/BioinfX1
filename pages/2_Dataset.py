import streamlit as st
import pandas as pd

st.title("📂 Force Field Dataset Explorer")

with open("forcefield_dataset.txt") as f:
    lines = f.readlines()

st.text_area("Dataset", "".join(lines), height=400)

# Convert to DataFrame (simple)
data = [line.split() for line in lines if not line.startswith("[") and line.strip()]
df = pd.DataFrame(data)

st.subheader("Preview")
st.dataframe(df.head(20))
