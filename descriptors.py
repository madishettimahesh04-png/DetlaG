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
