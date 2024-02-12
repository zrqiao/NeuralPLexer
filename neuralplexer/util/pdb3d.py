import json
import re
import subprocess
import sys
import tempfile
import warnings
from subprocess import check_output

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem

# https://bioinformatics.stackexchange.com/questions/14101/extract-residue-sequence-from-pdb-file-in-biopython-but-open-to-recommendation
d3to1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

OST_PATH = "/groups/tfm/zqiao/tools/openstructure/build/stage/bin"


def compute_residue_wise_lddt(model_pdb, ref_pdb):
    tmp_json = tempfile.NamedTemporaryFile().name
    # command = f"{OST_PATH}/ost compare-structures --model {model_pdb} --reference {ref_pdb} --output {tmp_json} --lddt --molck --remove oxt hyd unk nonstd --clean-element-column --map-nonstandard-residues --structural-checks --bond-tolerance 15.0 --angle-tolerance 15.0 --qs-score --inclusion-radius 15.0 -spr "
    # command = f"{OST_PATH}/ost compare-structures --model {model_pdb} --reference {ref_pdb} --output {tmp_json} --lddt --molck --remove oxt hyd unk nonstd --clean-element-column --map-nonstandard-residues --bond-tolerance 15.0 --angle-tolerance 15.0 --qs-score --inclusion-radius 15.0 -spr "
    command = f"{OST_PATH}/ost compare-structures --model {model_pdb} --reference {ref_pdb} --output {tmp_json} --lddt --molck --remove oxt hyd unk nonstd --clean-element-column --map-nonstandard-residues --bond-tolerance 15.0 --angle-tolerance 15.0 --qs-score --inclusion-radius 10.0 -spr "
    subprocess.run(
        command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    with open(tmp_json, "r") as tf:
        res = json.load(tf)
    return res


def get_aa_sequence(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_path)
    models = list(structure.get_models())
    chain = models[0][chain_id]
    return "".join([d3to1[residue.resname] for residue in chain])


def get_chainid(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_path)
    models = list(structure.get_models())
    model = models[0]
    return [chain.get_id() for chain in model]


def get_chain_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_path)
    models = list(structure.get_models())
    model = models[0]
    chain_coords = {}
    for chain in model:
        coords = []
        for res in chain:
            for atom in res:
                atom_pos = atom.get_coord()
                coords.append(atom_pos)
        chain_coords[chain.get_id()] = np.array(coords)
    return chain_coords


def compute_tm_rmsd(model, ref):
    tm_scores = []
    rmsds = []
    ret = check_output(["TMalign", model, ref])
    for line in str(ret).split("\\n"):
        if re.match(r"^Aligned", line):
            rmsds.append(float(line.split()[4][:-1]))  # Extract the value
        if re.match(r"^TM-score=", line):
            tm_scores.append(line.split()[1])  # Extract the value
    return max(tm_scores), min(rmsds)


def compute_ligand_rmsd(model, ref):
    ret = check_output(
        ["obrms", "-f", model, ref],
        stderr=subprocess.DEVNULL,
        encoding="UTF-8",
    )
    rmsd = None
    for line in str(ret).split("\n"):
        if re.match(r"^RMSD", line):
            rmsd = float(line.split()[2])
    return rmsd


def find_pdb_contact_chains(pdb1_chain_coords, pdb2, cutoff=4.0):
    mol2 = Chem.MolFromPDBFile(pdb2, sanitize=False, proximityBonding=False)
    xyz2 = np.array(mol2.GetConformer().GetPositions())
    binding_chains = []
    for chain_id in pdb1_chain_coords.keys():
        distmat = np.linalg.norm(
            pdb1_chain_coords[chain_id][None, :] - xyz2[:, None], axis=2
        )
        is_contact = distmat.min() <= cutoff
        if is_contact:
            binding_chains.append(chain_id)
    return binding_chains


def find_pdb_contact_residues(pdb1, pdb2, cutoff=4.0):
    mol2 = Chem.MolFromPDBFile(pdb2, sanitize=False, proximityBonding=False)
    xyz2 = np.array(mol2.GetConformer().GetPositions())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb1)
    models = list(structure.get_models())
    model = models[0]
    binding_site_residues = []
    for chain in model:
        seq, mask = [], []
        for res in chain:
            for atom in res:
                atom_pos = atom.get_coord()
                atom_to_lig_dist = np.linalg.norm(atom_pos[None, :] - xyz2, axis=1)
                is_contact = min(atom_to_lig_dist) <= cutoff
                if is_contact:
                    binding_site_residues.append(
                        (chain.get_id(), res.get_id()[1], res.get_resname())
                    )
                    break
    return binding_site_residues


def find_ligand_contact_residues(ref_pdb, mol, cutoff=4.0):
    lig_xyz = np.array(mol.GetConformer().GetPositions())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", ref_pdb)
    models = list(structure.get_models())
    model = models[0]

    binding_site_residues = []

    for chain in model:
        seq, mask = [], []
        for res in chain:
            for atom in res:
                atom_pos = atom.get_coord()
                atom_to_lig_dist = np.linalg.norm(atom_pos[None, :] - lig_xyz, axis=1)
                is_contact = min(atom_to_lig_dist) <= cutoff
                if is_contact:
                    binding_site_residues.append(
                        (chain.get_id(), res.get_id()[1], res.get_resname())
                    )
                    break
    return binding_site_residues


def compute_lddt_bs(pdb1, pdb2, lddt_dat, binding_site_residues, ignore_unmatched=True):
    pdb1_name = pdb1.split("/")[-1]
    pdb2_name = pdb2.split("/")[-1]
    lddt_dict = {}
    for entry in lddt_dat["result"][pdb1_name][pdb2_name]["lddt"]["single_chain_lddt"][
        0
    ]["per_residue_scores"]:
        res_name = entry["residue_name"]
        res_number = entry["residue_number"]
        res_lddt = entry["lddt"]
        lddt_dict[res_number] = (res_name, res_lddt)
    site_lddts = []
    for site in binding_site_residues:
        try:
            assert lddt_dict[site[1]][0] == site[2]
        except:
            if ignore_unmatched:
                continue
            else:
                raise
        site_lddts.append(lddt_dict[site[1]][1])
    return np.array(site_lddts)


def get_lddt_bs(pdb1, pdb2, sdf=None, mol=None):
    try:
        lddt_dat = compute_residue_wise_lddt(pdb1, pdb2)
        if mol is None:
            if sdf is None:
                return None
            fsuppl = Chem.SDMolSupplier(sdf, sanitize=False)
            mol = next(fsuppl)
        bs_res_list = find_ligand_contact_residues(pdb2, mol)
        lddt_site = compute_lddt_bs(pdb1, pdb2, lddt_dat, bs_res_list)
        return np.mean(lddt_site)
    except Exception as e:
        warnings.warn(f"lDDT calculation failed due to {e}")
        return None


def compute_nonconserved_q_factor(
    sample, alt_sample, pred_sample, contact_cutoff=4.0, min_motion_dist=0.0
):
    import torch

    from neuralplexer.data.pipeline import (process_template_protein_features,
                                            to_torch)

    batch_size = sample["metadata"]["num_structid"]
    sample = process_template_protein_features(sample, pred_sample, auto_align=True)
    pred_prot_coords = torch.tensor(
        sample["features"]["template_atom_positions"]
    ).clone()
    pred_alignment_mask = (
        torch.tensor(sample["features"]["template_alignment_mask"])
        .bool()
        .view(batch_size, -1)
        .clone()
    )
    sample = process_template_protein_features(sample, alt_sample, auto_align=True)
    sample = to_torch(sample)
    # assert torch.all(pred_sample["features"]["res_atom_mask"]==sample["features"]["res_atom_mask"])

    pred_cent_coords = (
        pred_prot_coords.mul(sample["features"]["res_atom_mask"].bool()[:, :, None])
        .sum(dim=1)
        .div(sample["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
    ).view(batch_size, -1, 3)
    # pred_cent_coords = pred_prot_coords[:, 1].view(batch_size, -1, 3)
    pred_dist = (
        torch.square(pred_cent_coords[:, :, None] - pred_cent_coords[:, None, :])
        .sum(-1)
        .add(1e-4)
        .sqrt()
        .sub(1e-2)
    )
    with torch.no_grad():
        target_cent_coords = (
            sample["features"]["res_atom_positions"]
            .mul(sample["features"]["res_atom_mask"].bool()[:, :, None])
            .sum(dim=1)
            .div(sample["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
        ).view(batch_size, -1, 3)
        # target_cent_coords = sample["features"]["res_atom_positions"][:, 1].view(batch_size, -1, 3)
        template_cent_coords = (
            sample["features"]["template_atom_positions"]
            .mul(sample["features"]["template_atom37_mask"].bool()[:, :, None])
            .sum(dim=1)
            .div(
                sample["features"]["template_atom37_mask"].bool().sum(dim=1)[:, None]
                + 1e-9
            )
        ).view(batch_size, -1, 3)
        # template_cent_coords = sample["features"]["template_atom_positions"][:, 1].view(batch_size, -1, 3)
        target_dist = (
            torch.square(
                target_cent_coords[:, :, None] - target_cent_coords[:, None, :]
            )
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
        template_dist = (
            torch.square(
                template_cent_coords[:, :, None] - template_cent_coords[:, None, :]
            )
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
        template_alignment_mask = (
            sample["features"]["template_alignment_mask"].bool().view(batch_size, -1)
        )
        # motion_mask = (
        #     ((template_dist-target_dist).abs()>min_motion_dist)
        #     * template_alignment_mask[:, None, :]
        #     * template_alignment_mask[:, :, None]
        # )
        # contact_cutoff = torch.median(target_dist[motion_mask])
        ref_contact_map = target_dist < contact_cutoff
        non_conserved_contact_mask = (
            ((template_dist < contact_cutoff) != (target_dist < contact_cutoff))
            * ((template_dist - target_dist).abs() > min_motion_dist)
            * template_alignment_mask[:, None, :]
            * template_alignment_mask[:, :, None]
        )
        # print(ref_contact_map.mul(non_conserved_contact_mask).sum(), non_conserved_contact_mask.sum())
    pred_contact_map = (
        (pred_dist < contact_cutoff)
        * pred_alignment_mask[:, None, :]
        * pred_alignment_mask[:, :, None]
    )
    nc_q_factor = (
        ((pred_contact_map == ref_contact_map) * non_conserved_contact_mask).sum(
            dim=(1, 2)
        )
    ) / (non_conserved_contact_mask.sum(dim=(1, 2)))
    return nc_q_factor


if __name__ == "__main__":
    pdb1 = sys.argv[1]
    pdb2 = sys.argv[2]
    sdf = sys.argv[3]
    print(get_lddt_bs(pdb1, pdb2, sdf=sdf))
