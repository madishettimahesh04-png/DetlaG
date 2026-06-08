import numpy as np
import pandas as pd

from rdkit import Chem

from rdkit.Chem import (
    Descriptors,
    Crippen,
    rdMolDescriptors,
    AllChem,
    Descriptors3D
)

# =================================================
# TOPOLOGICAL INDICES
# =================================================

def wiener_index(mol):

    dmat = Chem.GetDistanceMatrix(mol)

    n = dmat.shape[0]

    total = 0.0

    for i in range(n):

        for j in range(i + 1, n):

            total += dmat[i, j]

    return total


def zagreb_index(mol):

    return sum(
        atom.GetDegree() ** 2
        for atom in mol.GetAtoms()
    )


def randic_index(mol):

    val = 0.0

    for bond in mol.GetBonds():

        u = bond.GetBeginAtom().GetDegree()

        v = bond.GetEndAtom().GetDegree()

        if u > 0 and v > 0:

            val += 1.0 / np.sqrt(u * v)

    return val


def petitjean_index(mol):

    dmat = Chem.GetDistanceMatrix(mol)

    ecc = dmat.max(axis=1)

    diameter = ecc.max()

    radius = ecc.min()

    if diameter == 0:

        return 0.0

    return (diameter - radius) / diameter


# =================================================
# FUNCTIONAL GROUP SMARTS
# =================================================

FG_SMARTS = {

    "Carboxylic acid": "[CX3](=O)[OX2H1]",

    "Amide": "[NX3][CX3](=O)",

    "Ester": "[CX3](=O)[OX2][#6]",

    "Alcohol": "[OX2H][CX4]",

    "Amine": "[NX3;H2,H1;!$(NC=O)]",

    "Ketone": "[CX3](=O)[#6]",

    "Aldehyde": "[CX3H1](=O)",

    "Ether": "[OD2]([#6])[#6]",

    "Halide": "[F,Cl,Br,I]",
}


def detect_dominant_functional_group(mol):

    for fg, smarts in FG_SMARTS.items():

        patt = Chem.MolFromSmarts(smarts)

        if mol.HasSubstructMatch(patt):

            return fg

    return "Hydrocarbon"


# =================================================
# 3D CONFORMER GENERATION
# =================================================

def generate_3d_mol(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    mol = Chem.AddHs(mol)

    status = AllChem.EmbedMolecule(
        mol,
        AllChem.ETKDGv3()
    )

    if status != 0:

        return None

    try:

        if AllChem.MMFFHasAllMoleculeParams(mol):

            AllChem.MMFFOptimizeMolecule(mol)

        else:

            AllChem.UFFOptimizeMolecule(mol)

    except:
        pass

    return mol
    # =================================================
# MAIN DESCRIPTOR FUNCTION
# =================================================

def solvation_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {}

    logp = Crippen.MolLogP(mol)

    tpsa = rdMolDescriptors.CalcTPSA(mol)

    hbd = rdMolDescriptors.CalcNumHBD(mol)

    hba = rdMolDescriptors.CalcNumHBA(mol)

    polarity_class = (
        "Nonpolar" if tpsa < 20 else
        "Weakly polar" if tpsa < 40 else
        "Moderately polar" if tpsa < 75 else
        "Highly polar"
    )

    family = (
        "Polar protic" if hbd > 0 else
        "Polar aprotic" if hba > 0 else
        "Nonpolar"
    )

    hydrogen_class = (
        "Donor & Acceptor" if (hbd > 0 and hba > 0) else
        "Donor only" if hbd > 0 else
        "Acceptor only" if hba > 0 else
        "Non H-bonding"
    )

    desc = {

        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "LogP": logp,
        "TPSA": tpsa,
        "MolMR": Crippen.MolMR(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),

        "HBD": hbd,
        "HBA": hba,

        "HydrogenBondClass": hydrogen_class,
        "DominantFunctionalGroup": detect_dominant_functional_group(mol),
        "Family": family,
        "PolarityClass": polarity_class,
    }

    desc.update({

        "BalabanJ": Descriptors.BalabanJ(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "Chi0": Descriptors.Chi0(mol),
        "Chi1": Descriptors.Chi1(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),

        "WienerIndex": wiener_index(mol),
        "ZagrebIndex": zagreb_index(mol),
        "RandicIndex": randic_index(mol),
        "PetitjeanIndex": petitjean_index(mol),
    })

    mol3d = generate_3d_mol(smiles)

    if mol3d is None:

        desc.update({

            "MolVolume": 0,
            "LabuteASA": 0,
            "RadiusOfGyration": 0,
            "Asphericity": 0,
            "Eccentricity": 0,
            "InertialShapeFactor": 0,
            "SpherocityIndex": 0,
            "PMI1": 0,
            "PMI2": 0,
            "PMI3": 0,
            "PMI_ratio_1_2": 0,
            "PMI_ratio_2_3": 0,
        })

        return desc

    desc.update({

        "MolVolume":
            AllChem.ComputeMolVolume(mol3d),

        "LabuteASA":
            rdMolDescriptors.CalcLabuteASA(mol3d),

        "RadiusOfGyration":
            Descriptors3D.RadiusOfGyration(mol3d),

        "Asphericity":
            Descriptors3D.Asphericity(mol3d),

        "Eccentricity":
            Descriptors3D.Eccentricity(mol3d),

        "InertialShapeFactor":
            Descriptors3D.InertialShapeFactor(mol3d),

        "SpherocityIndex":
            Descriptors3D.SpherocityIndex(mol3d),

        "PMI1":
            Descriptors3D.PMI1(mol3d),

        "PMI2":
            Descriptors3D.PMI2(mol3d),

        "PMI3":
            Descriptors3D.PMI3(mol3d),

        "PMI_ratio_1_2":
            Descriptors3D.PMI1(mol3d) /
            (Descriptors3D.PMI2(mol3d) + 1e-6),

        "PMI_ratio_2_3":
            Descriptors3D.PMI2(mol3d) /
            (Descriptors3D.PMI3(mol3d) + 1e-6),
    })

    return desc
    # =================================================
# BUILD FINAL 155 FEATURES
# =================================================

def build_features(
    solute_smiles,
    solvent_smiles,
    feature_order
):

    solute = solvation_descriptors(
        solute_smiles
    )

    solvent = solvation_descriptors(
        solvent_smiles
    )

    features = {}

    # ==========================================
    # SOLUTE FEATURES
    # ==========================================

    for k, v in solute.items():

        features[f"solute_{k}"] = v

    # ==========================================
    # SOLVENT FEATURES
    # ==========================================

    for k, v in solvent.items():

        features[f"solvent_{k}"] = v

    # ==========================================
    # INTERACTION FEATURES
    # ==========================================

    numeric_keys = []

    for k, v in solute.items():

        if isinstance(
            v,
            (int, float, np.number)
        ):

            numeric_keys.append(k)

    for k in numeric_keys:

        if k not in solvent:
            continue

        try:

            s = float(solute[k])

            v = float(solvent[k])

            features[f"diff_{k}"] = abs(
                s - v
            )

            features[f"prod_{k}"] = (
                s * v
            )

            features[f"ratio_{k}"] = (
                s / (v + 1e-6)
            )

        except:
            pass

    # ==========================================
    # ONE HOT ENCODING
    # ==========================================

    categorical_cols = [

        "HydrogenBondClass",

        "DominantFunctionalGroup",

        "Family",

        "PolarityClass"
    ]

    for prefix, data in [

        ("solute", solute),

        ("solvent", solvent)

    ]:

        for cat in categorical_cols:

            if cat not in data:
                continue

            value = data[cat]

            col = (
                f"{prefix}_{cat}_{value}"
            )

            features[col] = 1

    # ==========================================
    # DATAFRAME
    # ==========================================

    X = pd.DataFrame(
        [features]
    )

    # ==========================================
    # MATCH TRAINED FEATURES
    # ==========================================

    for col in feature_order:

        if col not in X.columns:

            X[col] = 0

    X = X[feature_order]

    X = X.fillna(0)

    return X
