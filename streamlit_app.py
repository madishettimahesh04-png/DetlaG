import streamlit as st
import torch
import joblib

from rdkit import Chem

from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from model import Model
from descriptors import build_features

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="GNN Solubility Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Solubility Predictor")


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_all():

    cfg = joblib.load(
        "model_config.pkl"
    )

    pipeline = joblib.load(
        "pipeline.pkl"
    )

    feature_order = joblib.load(
        "feature_order.pkl"
    )

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

    return (
        model,
        pipeline,
        feature_order
    )

model, pipeline, feature_order = load_all()

scaler = pipeline["scaler"]

# =====================================================
# INPUTS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    solute = st.text_input(
        "Solute SMILES",
        value="CCO"
    )

with col2:

    solvent = st.text_input(
        "Solvent SMILES",
        value="O"
    )

# =====================================================
# PREDICT BUTTON
# =====================================================

predict_btn = st.button(
    "Predict ΔG"
)

# =====================================================
# PREDICTION
# =====================================================

if predict_btn:

    try:

        # -------------------------
        # Validate SMILES
        # -------------------------

        mol1 = Chem.MolFromSmiles(
            solute
        )

        mol2 = Chem.MolFromSmiles(
            solvent
        )

        if mol1 is None:

            st.error(
                "Invalid Solute SMILES"
            )

            st.stop()

        if mol2 is None:

            st.error(
                "Invalid Solvent SMILES"
            )

            st.stop()

        # -------------------------
        # Build 155 descriptors
        # -------------------------

        X = build_features(
            solute,
            solvent,
            feature_order
        )

        X_scaled = scaler.transform(
            X
        )

        desc_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        # -------------------------
        # Build Graphs
        # -------------------------

        g1 = from_smiles(
            solute
        )

        g2 = from_smiles(
            solvent
        )

        g1.x = g1.x.float()
        g2.x = g2.x.float()

        g1_batch = Batch.from_data_list(
            [g1]
        )

        g2_batch = Batch.from_data_list(
            [g2]
        )

        # -------------------------
        # Prediction
        # -------------------------

        with torch.no_grad():

            pred = model(
                g1_batch,
                g2_batch,
                desc_tensor
            )

        delta_g = float(pred)

        # -------------------------
        # Display Result
        # -------------------------

        st.success(
            f"Predicted ΔG = {delta_g:.4f}"
        )

        st.subheader("Input")

        st.write(
            {
                "Solute": solute,
                "Solvent": solvent
            }
        )

        st.subheader("Model Information")

        st.write(
            {
                "Graph Features": "GAT + GraphSAGE",
                "Descriptor Features": 155,
                "Target": "Solvation Free Energy (ΔG)"
            }
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )
