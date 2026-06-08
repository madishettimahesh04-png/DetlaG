from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

from model import Model

# =====================================================
# LOAD FILES
# =====================================================

cfg = joblib.load("model_config.pkl")

pipeline = joblib.load("pipeline.pkl")

feature_order = joblib.load("feature_order.pkl")

scaler = pipeline["scaler"]

# =====================================================
# LOAD MODEL
# =====================================================

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

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="GNN Solubility API"
)

# =====================================================
# INPUT
# =====================================================

class PredictionInput(BaseModel):
    solute_smiles: str
    solvent_smiles: str

# =====================================================
# RDKIT DESCRIPTORS
# =====================================================

def get_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    desc = {}

    desc["MolWt"] = Descriptors.MolWt(mol)
    desc["LogP"] = Crippen.MolLogP(mol)
    desc["TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    desc["MolMR"] = Crippen.MolMR(mol)

    desc["HeavyAtomCount"] = mol.GetNumHeavyAtoms()

    desc["NumRotatableBonds"] = (
        rdMolDescriptors.CalcNumRotatableBonds(mol)
    )

    desc["NumRings"] = (
        rdMolDescriptors.CalcNumRings(mol)
    )

    desc["NumAromaticRings"] = (
        rdMolDescriptors.CalcNumAromaticRings(mol)
    )

    desc["FracCSP3"] = (
        rdMolDescriptors.CalcFractionCSP3(mol)
    )

    desc["HBD"] = (
        rdMolDescriptors.CalcNumHBD(mol)
    )

    desc["HBA"] = (
        rdMolDescriptors.CalcNumHBA(mol)
    )

    return desc

# =====================================================
# BUILD FEATURES
# =====================================================

def build_features(solute_smiles, solvent_smiles):

    s = get_descriptors(solute_smiles)
    v = get_descriptors(solvent_smiles)

    features = {}

    # -----------------------------
    # solute
    # -----------------------------
    for k, val in s.items():
        features[f"solute_{k}"] = val

    # -----------------------------
    # solvent
    # -----------------------------
    for k, val in v.items():
        features[f"solvent_{k}"] = val

    # -----------------------------
    # interaction features
    # -----------------------------
    for k in s.keys():

        features[f"diff_{k}"] = (
            s[k] - v[k]
        )

        features[f"prod_{k}"] = (
            s[k] * v[k]
        )

        features[f"ratio_{k}"] = (
            s[k] / (v[k] + 1e-8)
        )

    # Fill missing training columns
    for col in feature_order:

        if col not in features:
            features[col] = 0

    X = pd.DataFrame([features])

    X = X[feature_order]

    return X

# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "message": "GNN API Running"
    }

# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
def predict(data: PredictionInput):

    try:

        X = build_features(
            data.solute_smiles,
            data.solvent_smiles
        )

        X_scaled = scaler.transform(X)

        desc_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        # -------------------------
        # graphs
        # -------------------------
        g1 = from_smiles(
            data.solute_smiles
        )

        g2 = from_smiles(
            data.solvent_smiles
        )

        g1.x = g1.x.float()
        g2.x = g2.x.float()

        g1_batch = Batch.from_data_list(
            [g1]
        )

        g2_batch = Batch.from_data_list(
            [g2]
        )

        with torch.no_grad():

            pred = model(
                g1_batch,
                g2_batch,
                desc_tensor
            )

        return {
            "solute": data.solute_smiles,
            "solvent": data.solvent_smiles,
            "prediction": float(pred)
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )