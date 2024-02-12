"""
Testing SMILES->tensorized features processing
"""

from neuralplexer.data.pipeline import (
    process_smiles,
    process_pdb,
    process_sdf,
    merge_protein_and_ligands,
)
from rdkit import Chem
from tqdm import tqdm
import pandas as pd


def test_processing_pdb():
    pdb_string = open("datasets/pdbbind/receptor_files/1a4w_protein_adfr.pdb").read()
    process_pdb(pdb_string)


def test_processing_sdf():
    sdf_suppl = Chem.SDMolSupplier("datasets/pdbbind/ligand_files/1a4w_ligand.mol")
    process_sdf(sdf_suppl, max_path_length=4, only_2d=True)


def test_merge_complex():
    sdf_suppl = Chem.SDMolSupplier("datasets/pdbbind/ligand_files/1a4w_ligand.mol")
    lig_sample = process_sdf(sdf_suppl, max_path_length=4, only_2d=True)
    pdb_string = open("datasets/pdbbind/receptor_files/1a4w_protein_adfr.pdb").read()
    rec_sample = process_pdb(pdb_string)
    merge_protein_and_ligands(lig_sample, rec_sample)
