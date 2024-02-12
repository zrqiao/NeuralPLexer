import sys
import os
import math
import argparse
import torch
import numpy as np
from neuralplexer.model.config import (
    get_base_config,
    _attach_binding_task_config,
    _attach_ligand_pretraining_task_config,
)
from deprecated import deprecated
from rdkit import Chem
from neuralplexer.model.common import segment_mean, segment_sum
from neuralplexer.model.wrappers import NeuralPlexer
from neuralplexer.data.pipeline import (
    process_pdb,
    process_mol_file,
    merge_protein_and_ligands,
    process_template_protein_features,
    to_torch,
    inplace_to_torch,
    inplace_to_cuda,
)
import tqdm
from neuralplexer.data.indexers import collate_numpy, collate_samples
from neuralplexer.data.dataloading import load_msgpack
from neuralplexer.util.frame import (
    RigidTransform,
    internal_to_cartesian,
    cartesian_to_internal,
    get_frame_matrix,
)
from neuralplexer.util.pdb3d import get_lddt_bs
from af_common.protein import Protein, from_prediction, to_pdb
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem.rdmolops import CombineMols
from rdkit.Chem.rdForceFieldHelpers import GetUFFVdWParams
import pandas as pd
import re
import subprocess
from subprocess import check_output
import multiprocessing as mp
import glob

torch.set_grad_enabled(False)


def get_sample_set(lig_path, rec_path, chain_id=None):
    if lig_path is not None:
        try:
            lig_sample, mol_ref = process_mol_file(
                lig_path,
                return_mol=True,
                pair_feats=True,
            )
        except:
            lig_sample, mol_ref = process_mol_file(
                lig_path,
                sanitize=False,
                return_mol=True,
                pair_feats=True,
            )
        rec_sample = process_pdb(open(rec_path).read(), chain_id=chain_id)
        merged_sample = merge_protein_and_ligands(
            [lig_sample],
            rec_sample,
            label=None,
            filter_ligands=False,
        )
        return merged_sample, mol_ref
    else:
        rec_sample = process_pdb(open(rec_path).read(), chain_id=chain_id)
        return rec_sample, None


if __name__ == "__main__":
    # Trivial
    sample, _ = get_sample_set(
        None, "datasets/all_pdb_parsed/010923_pdbfixer/7plzG.pdb"
    )
    template, _ = get_sample_set(
        None, "datasets/all_pdb_parsed/010923_pdbfixer/7plyC.pdb"
    )
    # Require alignment
    np.set_printoptions(threshold=sys.maxsize)
    sample, _ = get_sample_set(
        None, "datasets/all_pdb_parsed/010923_pdbfixer/5NCSB.pdb"
    )
    template, _ = get_sample_set(
        None, "datasets/all_pdb_parsed/010923_pdbfixer/5nctA.pdb"
    )
    for _ in tqdm.tqdm(range(100)):
        sample = process_template_protein_features(sample, template)
    print(sample["features"]["template_atom_positions"][:, 1])
    print(sample["features"]["template_alignment_mask"])
