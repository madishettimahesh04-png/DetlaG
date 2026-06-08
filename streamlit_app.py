import streamlit as st

st.title("GNN Solubility Predictor")

solute = st.text_input(
    "Solute SMILES",
    "CCO"
)

solvent = st.text_input(
    "Solvent SMILES",
    "O"
)

if st.button("Predict"):
    st.write("Prediction logic goes here")
