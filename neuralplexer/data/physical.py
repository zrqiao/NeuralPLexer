import os
from pathlib import Path

import numpy as np
import pandas as pd
from mendeleev.fetch import fetch_table

ptable = fetch_table("elements")
PATOMIC_NUMBERS = {row["symbol"]: row["atomic_number"] for _, row in ptable.iterrows()}
PGROUP_IDS = {row["symbol"]: row["group_id"] for _, row in ptable.iterrows()}
PPERIOD_IDS = {row["symbol"]: row["period"] for _, row in ptable.iterrows()}
VDW_RADII_MAPPING = {
    row["atomic_number"]: row["vdw_radius"] for _, row in ptable.iterrows()
}
UFF_VDW_RADII_MAPPING = {
    row["atomic_number"]: row["vdw_radius_uff"] for _, row in ptable.iterrows()
}


def get_vdw_radii_array():
    return np.array(
        [
            VDW_RADII_MAPPING[i] if i in VDW_RADII_MAPPING.keys() else -1.0
            for i in range(119)
        ]
    )


def get_vdw_radii_array_uff():
    return np.array(
        [
            UFF_VDW_RADII_MAPPING[i] if i in UFF_VDW_RADII_MAPPING.keys() else -1.0
            for i in range(119)
        ]
    )


uff_params_df = (
    pd.read_csv(
        os.path.join(
            Path(__file__).parent.absolute(),
            "uff_parameters.csv",
        ),
    )
    .drop_duplicates("Element")
    .set_index("Element")
)


def calc_heavy_atom_LJ_clash_fraction(mol1, mol2, LJ_CUTOFF=100):
    D_ij, x_ij = get_epsilon_sigma_uff(mol1, mol2)
    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())
    distance_matrix = np.linalg.norm(
        coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :],
        axis=2,
    )
    pairwise_LJ = D_ij * (
        (x_ij / distance_matrix) ** 12 - 2 * (x_ij / distance_matrix) ** 6
    )
    return np.sum(np.sum(pairwise_LJ, axis=1) > LJ_CUTOFF) / pairwise_LJ.shape[0]


def calc_heavy_atom_LJ_int_energy(mol1, mol2):
    D_ij, x_ij = get_epsilon_sigma_uff(mol1, mol2)
    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())
    distance_matrix = np.linalg.norm(
        coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :],
        axis=2,
    )
    pairwise_LJ = D_ij * (
        (x_ij / distance_matrix) ** 12 - 2 * (x_ij / distance_matrix) ** 6
    )
    return np.sum(pairwise_LJ)


def get_epsilon_sigma_uff(m1, m2):
    # Lorentz-Berthelot mixing
    Di = [
        float(uff_params_df.loc[atom.GetSymbol(), "energy"]) for atom in m1.GetAtoms()
    ]
    Dj = [
        float(uff_params_df.loc[atom.GetSymbol(), "energy"]) for atom in m2.GetAtoms()
    ]
    xi = [
        float(uff_params_df.loc[atom.GetSymbol(), "distance"]) for atom in m1.GetAtoms()
    ]
    xj = [
        float(uff_params_df.loc[atom.GetSymbol(), "distance"]) for atom in m2.GetAtoms()
    ]
    x_ij = (np.array(xi)[:, np.newaxis] + np.array(xj)[np.newaxis, :]) / 2
    D_ij = (np.array(Di)[:, np.newaxis] * np.array(Dj)[np.newaxis, :]) ** (1 / 2)
    return D_ij, x_ij
