import streamlit as st
import torch
import joblib
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from model import Model

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="GNN Solubility Predictor",
    layout="wide"
)

st.title("🧪 GNN Solubility Predictor")

# =====================================================
# LOAD FILES
# =====================================================

@st.cache_resource
def load_model():

    cfg = joblib.load("model_config.pkl")

    model = Model(
        desc_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        desc_hidden=cfg["desc_hidden"]
    )

    model.load_state_dict(
        torch.load(
            "best_model.pt",
            map_location="cpu"
        )
    )

    model.eval()

    return model

@st.cache_resource
def load_pipeline():

    return joblib.load("pipeline.pkl")

@st.cache_resource
def load_features():

    return joblib.load("feature_order.pkl")

model = load_model()
pipeline = load_pipeline()
feature_order = load_features()

# =====================================================
# INPUTS
# =====================================================

solute = st.text_input(
    "Solute SMILES",
    value="CCO"
)

solvent = st.text_input(
    "Solvent SMILES",
    value="O"
)

# =====================================================
# PREDICT
# =====================================================

if st.button("Predict"):

    try:

        mol1 = Chem.MolFromSmiles(solute)
        mol2 = Chem.MolFromSmiles(solvent)

        if mol1 is None:
            st.error("Invalid Solute SMILES")
            st.stop()

        if mol2 is None:
            st.error("Invalid Solvent SMILES")
            st.stop()

        g1 = from_smiles(solute)
        g2 = from_smiles(solvent)

        g1.x = g1.x.float()
        g2.x = g2.x.float()

        g1_batch = Batch.from_data_list([g1])
        g2_batch = Batch.from_data_list([g2])

        # ------------------------------------------------
        # TEMP PLACEHOLDER
        # Replace with descriptor generation later
        # ------------------------------------------------

        desc_tensor = torch.zeros(
            (1, 155),
            dtype=torch.float32
        )

        with torch.no_grad():

            pred = model(
                g1_batch,
                g2_batch,
                desc_tensor
            )

        st.success(
            f"Prediction: {float(pred):.4f}"
        )

    except Exception as e:

        st.error(str(e))
