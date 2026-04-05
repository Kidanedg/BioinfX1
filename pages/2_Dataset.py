import streamlit as st
import pandas as pd

st.title("📂 Force Field Dataset Explorer")

uploaded_file = st.file_uploader("Upload force field dataset")

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()

    st.text_area("Dataset", "\n".join(lines), height=400)

    data = [
        line.split()
        for line in lines
        if not line.startswith("[") and line.strip()
    ]

    df = pd.DataFrame(data)

    st.subheader("Preview")
    st.dataframe(df.head(20))
else:
    st.warning("Please upload a dataset file.")
