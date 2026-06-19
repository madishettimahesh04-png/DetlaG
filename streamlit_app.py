```python
import streamlit as st
import torch
import joblib
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from model import Model
from descriptors import build_features

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Solvation Free Energy Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Solvation Free Energy Predictor")
st.markdown(
    "Predict solvation free energy (ΔG) using a Hybrid GAT + GraphSAGE + Descriptor Model"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_all():

    cfg = joblib.load("model_config.pkl")

    pipeline = joblib.load("pipeline.pkl")

    feature_order = joblib.load("feature_order.pkl")

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

    return model, pipeline, feature_order


model, pipeline, feature_order = load_all()

scaler = pipeline["scaler"]

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_delta_g(solute, solvent):

    mol1 = Chem.MolFromSmiles(solute)

    mol2 = Chem.MolFromSmiles(solvent)

    if mol1 is None:
        raise ValueError("Invalid Solute SMILES")

    if mol2 is None:
        raise ValueError("Invalid Solvent SMILES")

    # -----------------------------
    # Descriptors
    # -----------------------------

    X = build_features(
        solute,
        solvent,
        feature_order
    )

    X_scaled = scaler.transform(X)

    desc_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    # -----------------------------
    # Graphs
    # -----------------------------

    g1 = from_smiles(solute)
    g2 = from_smiles(solvent)

    g1.x = g1.x.float()
    g2.x = g2.x.float()

    g1_batch = Batch.from_data_list([g1])
    g2_batch = Batch.from_data_list([g2])

    # -----------------------------
    # Prediction
    # -----------------------------

    with torch.no_grad():

        pred = model(
            g1_batch,
            g2_batch,
            desc_tensor
        )

    return float(pred)


# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "🔬 Single Prediction",
        "📂 Batch Prediction"
    ]
)

# =====================================================
# SINGLE PREDICTION
# =====================================================

with tab1:

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.subheader("Enter Molecules")

        solute = st.text_input(
            "Solute SMILES",
            value="CCO"
        )

        solvent = st.text_input(
            "Solvent SMILES",
            value="O"
        )

        predict_btn = st.button(
            "Predict ΔG",
            use_container_width=True
        )

        if predict_btn:

            try:

                delta_g = predict_delta_g(
                    solute,
                    solvent
                )

                st.success("Prediction Complete")

                st.metric(
                    label="Predicted ΔG",
                    value=f"{delta_g:.4f}"
                )

            except Exception as e:

                st.error(str(e))

# =====================================================
# BATCH PREDICTION
# =====================================================

with tab2:

    st.subheader("Upload CSV or Excel File")

    st.info(
        """
        Required columns:

        • Solute_SMILES

        • Solvent_SMILES
        """
    )

    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            # -------------------------
            # Read File
            # -------------------------

            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.subheader("Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            # -------------------------
            # Predict Button
            # -------------------------

            if st.button(
                "Run Batch Prediction"
            ):

                if (
                    "Solute_SMILES"
                    not in df.columns
                ):

                    st.error(
                        "Column 'Solute_SMILES' not found"
                    )

                    st.stop()

                if (
                    "Solvent_SMILES"
                    not in df.columns
                ):

                    st.error(
                        "Column 'Solvent_SMILES' not found"
                    )

                    st.stop()

                results = []

                progress_bar = st.progress(0)

                total_rows = len(df)

                for idx, row in df.iterrows():

                    try:

                        pred = predict_delta_g(
                            str(
                                row[
                                    "Solute_SMILES"
                                ]
                            ),
                            str(
                                row[
                                    "Solvent_SMILES"
                                ]
                            )
                        )

                        results.append(pred)

                    except:

                        results.append(None)

                    progress_bar.progress(
                        (idx + 1) / total_rows
                    )

                df["Predicted_DeltaG"] = results

                st.success(
                    "Batch Prediction Complete"
                )

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                csv = df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )

                st.download_button(
                    label="⬇ Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )
```
