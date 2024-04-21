import argparse
import glob
import math
import multiprocessing as mp
import os
import re
import subprocess
import warnings
from subprocess import check_output

import numpy as np
import pandas as pd
import torch
import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from af_common.residue_constants import restype_1to3
from neuralplexer.data.indexers import collate_numpy
from neuralplexer.data.physical import calc_heavy_atom_LJ_clash_fraction
from neuralplexer.data.pipeline import (featurize_protein_and_ligands,
                                        inplace_to_cuda, inplace_to_torch,
                                        process_mol_file, write_conformer_sdf,
                                        write_pdb_models, write_pdb_single)
from neuralplexer.model.config import (_attach_binding_task_config,
                                       get_base_config)
from neuralplexer.model.wrappers import NeuralPlexer
from neuralplexer.util.pdb3d import (compute_ligand_rmsd, compute_tm_rmsd,
                                     get_lddt_bs)

torch.set_grad_enabled(False)


def create_full_pdb_with_zero_coordinates(sequence: str, filename) -> None:
    """
    Create a PDB file with all atom coordinates set to zero for given protein sequences,
    including all atoms (backbone and simplified side chain). Multiple protein chains are
    delimited by "|".

    Args:
    sequence (str): Protein sequences in single-letter code, separated by "|".
    filename (str): Output file name for the PDB file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Backbone atoms for all amino acids
    backbone_atoms = ["N", "CA", "C", "O"]

    # Simplified representation of side chain atoms for each amino acid
    side_chain_atoms = {
        "A": ["CB"],
        "R": ["CB", "CG", "CD", "NE", "CZ"],
        "N": ["CB", "CG", "OD1"],
        "D": ["CB", "CG", "OD1"],
        "C": ["CB", "SG"],
        "E": ["CB", "CG", "CD"],
        "Q": ["CB", "CG", "CD", "OE1"],
        "G": [],
        "H": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
        "I": ["CB", "CG1", "CG2", "CD1"],
        "L": ["CB", "CG", "CD1", "CD2"],
        "K": ["CB", "CG", "CD", "CE", "NZ"],
        "M": ["CB", "CG", "SD", "CE"],
        "F": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "P": ["CB", "CG", "CD"],
        "S": ["CB", "OG"],
        "T": ["CB", "OG1", "CG2"],
        "W": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "Y": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "V": ["CB", "CG1", "CG2"],
    }

    with open(filename, "w") as pdb_file:
        atom_index = 1
        chain_id = "A"  # Start with chain 'A'

        for chain in sequence.split("|"):
            residue_index = 1
            for residue in chain:
                # Add backbone atoms
                for atom in backbone_atoms:
                    pdb_file.write(
                        f"ATOM  {atom_index:5d}  {atom:<3s} {restype_1to3.get(residue, 'UNK')} {chain_id}{residue_index:4d}    "
                        f"   0.000   0.000   0.000  1.00  0.00           C\n"
                    )
                    atom_index += 1

                # Add side chain atoms
                for atom in side_chain_atoms.get(residue, []):  # type: ignore
                    pdb_file.write(
                        f"ATOM  {atom_index:5d}  {atom:<3s} {restype_1to3.get(residue, 'UNK')} {chain_id}{residue_index:4d}    "
                        f"   0.000   0.000   0.000  1.00  0.00           C\n"
                    )
                    atom_index += 1

                residue_index += 1
            # Increment chain ID for next chain
            chain_id = chr(ord(chain_id) + 1)


def single_sample_sampling(args, model):
    sample, mol = featurize_protein_and_ligands(
        args.input_ligand,
        args.input_receptor,
        n_lig_patches=model.config.mol_encoder.n_patches,
        template_path=args.input_template,
    )
    sample = inplace_to_torch(collate_numpy([sample]))
    if args.cuda:
        sample = inplace_to_cuda(sample)

    all_frames = model.sample_pl_complex_structures(
        sample,
        sampler=args.sampler,
        num_steps=args.num_steps,
        return_all_states=True,
        start_time=args.start_time,
        exact_prior=args.exact_prior,
    )
    struct_res_all, lig_res_all = [], []
    for t, output_struct in enumerate(all_frames):
        out_x2 = output_struct["receptor_padded"].cpu().numpy()
        struct_res = {
            "features": {
                "asym_id": sample["features"]["res_chain_id"].long().cpu().numpy(),
                "residue_index": np.arange(len(sample["features"]["res_type"])) + 1,
                "aatype": sample["features"]["res_type"].long().cpu().numpy(),
            },
            "structure_module": {
                "final_atom_positions": out_x2.copy(),
                "final_atom_mask": sample["features"]["res_atom_mask"]
                .bool()
                .cpu()
                .numpy(),
            },
        }
        struct_res_all.append(struct_res)
        if mol is not None:
            lig_res_all.append(output_struct["ligands"].cpu().numpy())
    write_pdb_models(struct_res_all, out_path=args.out_path + "_prot.pdb")
    write_pdb_models([struct_res_all[-1]], out_path=args.out_path + "_prot_final.pdb")
    if mol is not None:
        write_conformer_sdf(
            mol, np.array([lig_res_all[-1]]), out_path=args.out_path + "_lig_final.sdf"
        )
        lig_res_all = np.array(lig_res_all)
        write_conformer_sdf(mol, lig_res_all, out_path=args.out_path + "_lig.sdf")

    return 0


def multi_pose_sampling(
    ligand_path,
    receptor_path,
    args,
    model,
    out_path,
    save_pdb=True,
    separate_pdb=True,
    chain_id=None,
    template_path=None,
    confidence=True,
    **kwargs,
):
    struct_res_all, lig_res_all = [], []
    plddt_all, plddt_lig_all, res_plddt_all = [], [], []
    chunk_size = args.chunk_size
    for _ in range(args.n_samples // chunk_size):
        # Resample anchor node frames
        np_sample, mol = featurize_protein_and_ligands(
            ligand_path,
            receptor_path,
            n_lig_patches=model.config.mol_encoder.n_patches,
            chain_id=chain_id,
            template_path=template_path,
            discard_sdf_coords=args.discard_sdf_coords,
            **kwargs,
        )
        np_sample_batched = collate_numpy([np_sample for _ in range(chunk_size)])
        sample = inplace_to_torch(np_sample_batched)
        if args.cuda:
            sample = inplace_to_cuda(sample)
        output_struct = model.sample_pl_complex_structures(
            sample,
            sampler=args.sampler,
            num_steps=args.num_steps,
            return_all_states=False,
            start_time=args.start_time,
            exact_prior=args.exact_prior,
        )
        if mol is not None:
            ref_mol = AllChem.Mol(mol)
            out_x1 = np.split(output_struct["ligands"].cpu().numpy(), args.chunk_size)
        out_x2 = np.split(
            output_struct["receptor_padded"].cpu().numpy(), args.chunk_size
        )
        if confidence:
            plddt, plddt_lig = model.run_confidence_estimation(
                sample, output_struct, return_avg_stats=True
            )
            res_plddt_all.append(
                sample["outputs"]["plddt"][
                    struct_idx, : sample["metadata"]["num_a_per_sample"][0]
                ]
                .cpu()
                .numpy()
            )

        for struct_idx in range(args.chunk_size):
            struct_res = {
                "features": {
                    "asym_id": np_sample["features"]["res_chain_id"],
                    "residue_index": np.arange(len(np_sample["features"]["res_type"]))
                    + 1,
                    "aatype": np_sample["features"]["res_type"],
                },
                "structure_module": {
                    "final_atom_positions": out_x2[struct_idx],
                    "final_atom_mask": sample["features"]["res_atom_mask"]
                    .bool()
                    .cpu()
                    .numpy(),
                },
            }
            struct_res_all.append(struct_res)
            if mol is not None:
                lig_res_all.append(out_x1[struct_idx])
            if confidence:
                plddt_all.append(plddt[struct_idx].item())
                if plddt_lig is None:
                    plddt_lig_all.append(None)
                else:
                    plddt_lig_all.append(plddt_lig[struct_idx].item())
    if confidence and args.rank_outputs_by_confidence:
        struct_plddts = np.array(plddt_lig_all if all(plddt_lig_all) else plddt_all)  # rank outputs using ligand plDDT if available
        struct_plddt_rankings = np.argsort(-struct_plddts).argsort()  # ensure that higher plDDTs have a higher rank (e.g., `rank1`)
    if save_pdb:
        receptor_plddt = np.array(res_plddt_all) if confidence else None
        b_factors = np.repeat(
            receptor_plddt[..., None],
            struct_res_all[0]["structure_module"]["final_atom_mask"].shape[-1],
            axis=-1,
        ) if confidence else None
        if separate_pdb:
            for struct_id, struct_res in enumerate(struct_res_all):
                if confidence and args.rank_outputs_by_confidence:
                    write_pdb_single(
                        struct_res, out_path=os.path.join(out_path, f"prot_rank{struct_plddt_rankings[struct_id] + 1}_plddt{struct_plddts[struct_id]:.4f}.pdb", b_factors=b_factors[struct_id] if confidence else None)
                    )
                else:
                    write_pdb_single(
                        struct_res, out_path=os.path.join(out_path, f"prot_{struct_id}.pdb"), b_factors=b_factors[struct_id] if confidence else None
                    )
        write_pdb_models(
            struct_res_all, out_path=os.path.join(out_path, f"prot_all.pdb"), b_factors=b_factors
        )
    if mol is not None:
        write_conformer_sdf(
            ref_mol, None, out_path=os.path.join(out_path, f"lig_ref.sdf")
        )
        lig_res_all = np.array(lig_res_all)
        write_conformer_sdf(
            mol, lig_res_all, out_path=os.path.join(out_path, f"lig_all.sdf")
        )
        for struct_id in range(len(lig_res_all)):
            if confidence and args.rank_outputs_by_confidence:
                write_conformer_sdf(
                    mol,
                    lig_res_all[struct_id : struct_id + 1],
                    out_path=os.path.join(out_path, f"lig_rank{struct_plddt_rankings[struct_id] + 1}_plddt{struct_plddts[struct_id]:.4f}.sdf"),
                )
            else:
                write_conformer_sdf(
                    mol,
                    lig_res_all[struct_id : struct_id + 1],
                    out_path=os.path.join(out_path, f"lig_{struct_id}.sdf"),
                )
    else:
        ref_mol = None
    if confidence:
        return ref_mol, plddt_all, plddt_lig_all
    return ref_mol


def pdbbind_benchmarking(args, model):
    """Rigid-protein flexible ligand docking"""
    args.start_time = 1.0
    args.exact_prior = False
    df = pd.read_csv(args.csv_path).replace({np.nan: None})
    df = df[df["test"] == True]
    res = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        code = row["pdb_id"]
        lig_path = f"{args.input_ligand}/{code}/ligand.sdf"
        rec_path = f"{args.input_receptor}/{code}/protein.pdb"
        out_path = os.path.join(args.out_path, code)
        os.makedirs(out_path, exist_ok=True)
        try:
            _, plddt, plddt_lig = multi_pose_sampling(
                [lig_path],
                rec_path,
                args,
                model,
                out_path,
                template_path="self",
                confidence=True,
                enforce_sanitization=True,
                allow_insertion_code=True,
            )
        except:
            lig_path = lig_path[:-4] + ".mol2"
            _, plddt, plddt_lig = multi_pose_sampling(
                [lig_path],
                rec_path,
                args,
                model,
                out_path,
                template_path="self",
                confidence=True,
                enforce_sanitization=True,
                allow_insertion_code=True,
            )
        try:
            mol_ref = process_mol_file(lig_path, sanitize=True, featurize=False)
            nrotbonds = AllChem.CalcNumRotatableBonds(mol_ref, strict=True)
            geom_ref = np.array(mol_ref.GetConformer().GetPositions())
            write_conformer_sdf(
                mol_ref, np.array([geom_ref]), out_path=out_path + "/lig_ref.sdf"
            )
            out_sdf_file = out_path + "/lig_all.sdf"
            ret = check_output(
                ["obrms", "-f", out_path + "/lig_ref.sdf", out_sdf_file],
                stderr=subprocess.DEVNULL,
                encoding="UTF-8",
            )
            RMSDlist = []
            for line in str(ret).split("\n"):
                if re.match(r"^RMSD", line):
                    rmsd = float(line.split()[2])
                    RMSDlist.append(rmsd)
            assert len(RMSDlist) == args.n_samples
            CENTDlist = []
            fsuppl = AllChem.SDMolSupplier(out_sdf_file)
            for i in tqdm.tqdm(range(args.n_samples), desc="Computing metrics"):
                mol_pred = next(fsuppl)
                cent_ref = np.mean(geom_ref, axis=0)
                geom_pred = np.array(mol_pred.GetConformer().GetPositions())
                cent_pred = np.mean(geom_pred, axis=0)
                cent_d = np.linalg.norm(cent_ref - cent_pred)
                CENTDlist.append(cent_d)
            Clashlist = []
            prot_ref = AllChem.MolFromPDBFile(rec_path, sanitize=False)
            prot_ref = AllChem.RemoveHs(prot_ref, sanitize=False)
            fsuppl = AllChem.SDMolSupplier(out_sdf_file)
            clash_gt = calc_heavy_atom_LJ_clash_fraction(mol_ref, prot_ref)
            for i in range(args.n_samples):
                mol_pred = next(fsuppl)
                mol_pred = AllChem.RemoveHs(mol_pred, sanitize=False)
                Clashlist.append(calc_heavy_atom_LJ_clash_fraction(mol_pred, prot_ref))
            res.append(
                [code, nrotbonds]
                + RMSDlist
                + CENTDlist
                + [clash_gt]
                + Clashlist
                + plddt
                + plddt_lig
            )
        except Exception as e:
            warnings.warn(f"Sample {code} failed: {e}")
            continue
        res_df = pd.DataFrame(
            columns=["index", "nrotbonds"]
            + [f"rmsd_{i}" for i in range(args.n_samples)]
            + [f"centd_{i}" for i in range(args.n_samples)]
            + ["clash_gt"]
            + [f"clash_{i}" for i in range(args.n_samples)]
            + [f"plddt_{i}" for i in range(args.n_samples)]
            + [f"plddt_lig_{i}" for i in range(args.n_samples)],
            data=res,
        )
        res_df.to_csv(f"{args.out_path}/pdbbind2019+_benchmarking_summary.csv")


def protein_inpainting_benchmarking(args, model):
    args.start_time = 1.0
    args.exact_prior = False
    os.makedirs(args.out_path, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    res = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        sample_id = row["sample_id"]
        out_path = os.path.join(args.out_path, sample_id)
        os.makedirs(out_path, exist_ok=True)
        lig_path = f"{args.input_ligand}/{sample_id}/ligand.sdf"
        ref_pdb = f"{args.input_receptor}/{sample_id}/protein.pdb"
        template_path = f"{args.input_template}/{sample_id}/af2_template.pdb"
        try:
            try:
                ref_mol = multi_pose_sampling(
                    [lig_path],
                    template_path,
                    args,
                    model,
                    out_path,
                    template_path=template_path,
                    allow_insertion_code=True,
                    confidence=False,
                )
            except:
                lig_path = lig_path[:-4] + ".mol2"
                ref_mol = multi_pose_sampling(
                    [lig_path],
                    template_path,
                    args,
                    model,
                    out_path,
                    template_path=template_path,
                    allow_insertion_code=True,
                    confidence=False,
                )
            lddt_bs_list = []
            tm_scores = []
            rmsds = []
            for i in tqdm.tqdm(range(args.n_samples), desc="Computing metrics"):
                in_pdb = os.path.join(out_path, f"prot_{i}.pdb")
                tm_score, tm_rmsd = compute_tm_rmsd(in_pdb, ref_pdb)
                tm_scores.append(tm_score)
                rmsds.append(tm_rmsd)
                lddt_bs = get_lddt_bs(in_pdb, ref_pdb, mol=ref_mol)
                lddt_bs_list.append(lddt_bs)
            lddt_bs_af2 = get_lddt_bs(template_path, ref_pdb, mol=ref_mol)

            out_sdf_file = os.path.join(out_path, f"lig_all.sdf")
            ret = check_output(
                ["obrms", "-f", lig_path, out_sdf_file],
                stderr=subprocess.DEVNULL,
                encoding="UTF-8",
            )
            RMSDlist = []
            for line in str(ret).split("\n"):
                if re.match(r"^RMSD", line):
                    rmsd = float(line.split()[2])
                    RMSDlist.append(rmsd)
            assert len(rmsds) == args.n_samples
            fsuppl = AllChem.SDMolSupplier(lig_path, sanitize=False)
            mol_ref = next(fsuppl)
            mol_ref = AllChem.RemoveHs(mol_ref, sanitize=False)
            np.array(mol_ref.GetConformer().GetPositions())
            Clashlist = []
            prot_ref = AllChem.MolFromPDBFile(ref_pdb, sanitize=False)
            prot_ref = AllChem.RemoveHs(prot_ref, sanitize=False)
            prot_af2 = AllChem.MolFromPDBFile(template_path, sanitize=False)
            prot_af2 = AllChem.RemoveHs(prot_af2, sanitize=False)
            fsuppl = AllChem.SDMolSupplier(out_sdf_file, sanitize=False)
            clash_gt = calc_heavy_atom_LJ_clash_fraction(mol_ref, prot_ref)
            clash_af2 = calc_heavy_atom_LJ_clash_fraction(mol_ref, prot_af2)
            for i in range(args.n_samples):
                in_pdb = os.path.join(out_path, f"prot_{i}.pdb")
                mol_pred = next(fsuppl)
                mol_pred = AllChem.RemoveHs(mol_pred, sanitize=False)
                prot_pred = AllChem.MolFromPDBFile(in_pdb, sanitize=False)
                prot_pred = AllChem.RemoveHs(prot_pred, sanitize=False)
                Clashlist.append(calc_heavy_atom_LJ_clash_fraction(mol_pred, prot_pred))
            res.append(
                [sample_id, in_pdb]
                + tm_scores
                + lddt_bs_list
                + [lddt_bs_af2]
                + RMSDlist
                + [clash_gt, clash_af2]
                + Clashlist
            )
        except Exception as e:
            warnings.warn(f"Sample {sample_id} failed: {e}")
            continue
        res_df = pd.DataFrame(
            columns=["sample_id", "filename"]
            + [f"TMScore_{i}" for i in range(args.n_samples)]
            + [f"lDDT-BS_{i}" for i in range(args.n_samples)]
            + ["lDDT-BS_af2"]
            + [f"lig_rmsd_{i}" for i in range(args.n_samples)]
            + ["clash_gt", "clash_af2"]
            + [f"clash_{i}" for i in range(args.n_samples)],
            data=res,
        )
        res_df.to_csv(
            f"{args.out_path}/PDBBind2019+bychain_AF2selected_benchmarking_summary.csv"
        )


def pl_structure_prediction_benchmarking(args, model):
    os.makedirs(args.out_path, exist_ok=True)
    df = pd.read_csv(args.csv_path).replace({np.nan: None})
    res = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        sample_id = row["protein_sample_id"]
        rec_path = f"{args.input_receptor}/{sample_id}.pdb"
        if args.discard_ligand or (not row["ligands_sample_id"]):
            lig_paths = []
        else:
            lig_paths = [
                f"{args.input_ligand}/{lig_id}.sdf"
                for lig_id in row["ligands_sample_id"].split(",")
            ]
        out_path = os.path.join(args.out_path, sample_id)
        os.makedirs(out_path, exist_ok=True)
        try:
            T_max = float(args.start_time)
            args.start_time = T_max
            if model.config.task.use_template:
                openfold_id = row["openfold_id"]
                template_query = f"{args.input_template}/{openfold_id}_unrelaxed_*_model_{args.template_id}.pdb"
                template_path = glob.glob(template_query)[0]
                print(f"Using template: {template_path}")
            else:
                template_path = None
            # NOTE: no chain id as all chains are relabelled to A by pdbfixer
            ref_mol, plddt, plddt_lig = multi_pose_sampling(
                lig_paths,
                rec_path,
                args,
                model,
                out_path,
                template_path=template_path,
                allow_insertion_code=True,
                confidence=True,
            )
            for run_idx in tqdm.tqdm(range(args.n_samples), desc="Computing metrics"):
                pred_pdb_path = os.path.join(out_path, f"prot_{run_idx}.pdb")
                tm_score, tm_rmsd = compute_tm_rmsd(pred_pdb_path, rec_path)
                lddt_bs = get_lddt_bs(pred_pdb_path, rec_path, mol=ref_mol)
                if template_path is not None:
                    tm_score_template, rmsd_template = compute_tm_rmsd(
                        pred_pdb_path, template_path
                    )
                    template_ref_tm, template_ref_rmsd = compute_tm_rmsd(
                        template_path, rec_path
                    )
                    template_lddt_bs = get_lddt_bs(template_path, rec_path, mol=ref_mol)
                else:
                    tm_score_template, rmsd_template = None, None
                    template_ref_tm, template_ref_rmsd = None, None
                    template_lddt_bs = None
                ref_lig_path = os.path.join(out_path, f"lig_ref.sdf")
                pred_lig_path = os.path.join(out_path, f"lig_{run_idx}.sdf")
                lig_rmsd = compute_ligand_rmsd(pred_lig_path, ref_lig_path)
                res.append(
                    [
                        sample_id,
                        run_idx,
                        pred_pdb_path,
                        rec_path.split("/")[-1],
                        ",".join([p.split("/")[-1] for p in lig_paths]),
                        tm_score,
                        tm_rmsd,
                        lig_rmsd,
                        lddt_bs,
                        plddt[run_idx],
                        plddt_lig[run_idx],
                        template_path.split("/")[-1] if template_path else None,
                        tm_score_template,
                        rmsd_template,
                        template_ref_tm,
                        template_ref_rmsd,
                        template_lddt_bs,
                    ]
                )
        except Exception as e:
            warnings.warn(f"Sample {sample_id} failed: {e}")
            continue

        res_df = pd.DataFrame(
            columns=[
                "sample_id",
                "run_id",
                "filename",
                "ref_pdb",
                "ref_sdf",
                "TMScore",
                "Calpha_rmsd",
                "ligand_rmsd",
                "lDDT-BS",
                "plddt",
                "plddt_lig",
                "template_pdb",
                "TMScore_pred_template",
                "RMSD_pred_template",
                "TMScore_template_ref",
                "RMSD_template_ref",
                "lDDT-BS_template_ref",
            ],
            data=res,
        )
        res_df.to_csv(os.path.join(args.out_path, "summary.csv"))
    print("Summary statistics:", res_df.quantile([0.25, 0.5, 0.75], numeric_only=True))


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--sample-id", default=0, type=int)
    parser.add_argument("--template-id", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--model-checkpoint", type=str)
    parser.add_argument("--input-ligand", type=str)
    parser.add_argument("--input-receptor", type=str)
    parser.add_argument("--input-template", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--n-samples", default=64, type=int)
    parser.add_argument("--chunk-size", default=8, type=int)
    parser.add_argument("--num-steps", default=100, type=int)
    parser.add_argument("--latent-model", type=str)
    parser.add_argument("--sampler", required=True, type=str)
    parser.add_argument("--start-time", default="1.0", type=str)
    parser.add_argument("--max-chain-encoding-k", default=-1, type=int)
    parser.add_argument("--exact-prior", action="store_true")
    parser.add_argument("--discard-ligand", action="store_true")
    parser.add_argument("--discard-sdf-coords", action="store_true")
    parser.add_argument("--detect-covalent", action="store_true")
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--separate-pdb", action="store_true")
    parser.add_argument("--rank-outputs-by-confidence", action="store_true")
    parser.add_argument("--csv-path", type=str)
    args = parser.parse_args()
    config = get_base_config()

    if args.model_checkpoint is not None:
        # No need to specify this when loading the entire model
        model = NeuralPlexer.load_from_checkpoint(
            checkpoint_path=args.model_checkpoint, strict=False
        )
        config = model.config
        if args.latent_model is not None:
            config.latent_model = args.latent_model
        if args.task == "pdbbind_benchmarking":
            config.task.use_template = True
        elif args.task == "binding_site_recovery_benchmarking":
            config.task.use_template = True
        else:
            config.task.use_template = args.use_template
        config.task.detect_covalent = args.detect_covalent

    model = NeuralPlexer.load_from_checkpoint(
        config=config, checkpoint_path=args.model_checkpoint, strict=False
    )
    model.eval()
    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        model.cuda()

    if args.start_time != "auto":
        args.start_time = float(args.start_time)
    if args.task == "single_sample_trajectory":
        single_sample_sampling(args, model)
    elif args.task == "batched_structure_sampling":
        # Handle no ligand input
        if args.input_ligand is not None:
            ligand_paths = list(args.input_ligand.split("|"))
        else:
            ligand_paths = None
        if not args.input_receptor.endswith(".pdb"):
            warnings.warn("Assuming the provided receptor input is a protein sequence")
            create_full_pdb_with_zero_coordinates(
                args.input_receptor, args.out_path + "/input.pdb"
            )
            args.input_receptor = args.out_path + "/input.pdb"
        multi_pose_sampling(
            ligand_paths,
            args.input_receptor,
            args,
            model,
            args.out_path,
            template_path=args.input_template,
            separate_pdb=args.separate_pdb,
        )
    elif args.task == "structure_prediction_benchmarking":
        pl_structure_prediction_benchmarking(args, model)
    elif args.task == "pdbbind_benchmarking":
        pdbbind_benchmarking(args, model)
    elif args.task == "binding_site_recovery_benchmarking":
        protein_inpainting_benchmarking(args, model)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
