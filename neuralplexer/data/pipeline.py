"""
Prepare tensorized model inputs from molecular graphs.
"""

import random
import warnings

import msgpack
import msgpack_numpy as m
import numpy as np
import torch
from deprecated import deprecated
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D

from af_common.protein import Protein
from af_common.protein import from_pdb_string as af_protein_from_pdb_string
from af_common.protein import from_prediction, to_pdb
from neuralplexer.data.indexers import collate_numpy, tensorize_indexers
from neuralplexer.data.molops import (compute_all_stereo_chemistry_encodings,
                                      compute_bond_pair_triangles,
                                      get_atom_encoding, get_bond_encoding,
                                      get_conformers_as_tensor, mol_to_graph)

m.patch()


def process_smiles(smiles: str, **kwargs):
    mol = Chem.MolFromSmiles(smiles)
    return _process_molecule(mol, **kwargs)


def _get_empty_mol_feature_set():
    metadata = {
        "num_molid": 0,
        "num_i": 0,
        "num_j": 0,
        "num_k": 0,
        # "num_u0": 0,
        "num_u": 0,
        "num_uv": 0,
        "num_ij": 0,
        "num_jk": 0,
        "num_ijk": 0,
        "num_jkl": 0,
        "num_ijkl": 0,
        # "num_u0ijk": 0,
        "num_Uijk": 0,
        "num_ujkl": 0,
    }
    indexer = tensorize_indexers(
        [],
        [],
        [],
        [],
        [],
        [],
        allow_dummy=True,
    )
    indexer["gather_idx_i_molid"] = np.zeros((0,), dtype=np.int_)
    indexer["gather_idx_ijk_molid"] = np.zeros((0,), dtype=np.int_)
    features = {
        "atom_encodings": np.zeros((0, 23), dtype=np.int_),
        "bond_encodings": np.zeros((0, 4), dtype=np.int_),
        "atom_positional_encodings": np.zeros((0, 6), dtype=np.int_),
        "stereo_chemistry_encodings": np.zeros((0, 14), dtype=np.int_),
        "atomic_numbers": np.zeros((0,), dtype=np.int_),
        "sdf_coordinates": np.zeros((0, 3), dtype=np.float_),
    }
    return {
        "metadata": metadata,
        "indexer": indexer,
        "features": features,
        "misc": {},
    }


def process_mol_file(
    fname,
    featurize=True,
    tokenizer=None,
    return_mol=False,
    sanitize=True,
    coord_feats=True,
    pair_feats=False,
    discard_coords=False,
    **kwargs,
):
    if fname.endswith("msgpack"):
        with open(fname, "rb") as data_file:
            byte_data = data_file.read()
            data_loaded = msgpack.unpackb(byte_data)
        return data_loaded
    elif fname.endswith("sdf"):
        fsuppl = Chem.SDMolSupplier(fname, sanitize=sanitize)
        mol = next(fsuppl)
        if not sanitize:
            mol.UpdatePropertyCache(strict=False)
    elif fname.endswith("mol2"):
        mol = Chem.MolFromMol2File(fname, sanitize=sanitize)
    elif fname.endswith("pdb"):
        mol = Chem.MolFromPDBFile(fname, sanitize=sanitize)
        if not sanitize:
            mol.UpdatePropertyCache(strict=False)
    else:
        warnings.warn(f"No suffix found for ligand input, assuming SMILES input")
        mol = Chem.MolFromSmiles(fname, sanitize=sanitize)
    if sanitize:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
    else:
        mol = Chem.RemoveHs(mol, sanitize=False)
    if not featurize:
        return mol
    if tokenizer is not None:
        return smiles2inputs(tokenizer, Chem.rdmolfiles.MolToSmiles(mol))
    if discard_coords:
        conf_xyz = get_conformers_as_tensor(mol, 1)[0]
    else:
        conf_xyz = np.array(mol.GetConformer().GetPositions())
    metadata, indexer, features = _process_molecule(
        mol, return_mol=False, ref_conf_xyz=conf_xyz, **kwargs
    )
    if coord_feats:
        features["sdf_coordinates"] = conf_xyz
    feature_set = {
        "metadata": metadata,
        "indexer": indexer,
        "features": features,
        "misc": {},
    }
    if pair_feats:
        _attach_pair_idx_and_encodings(feature_set, **kwargs)
    if return_mol:
        return feature_set, mol
    return feature_set


def _attach_pair_idx_and_encodings(feature_set, max_n_frames=None, lazy_eval=False):
    if lazy_eval:
        if "scatter_idx_u0ijk_Uijk" in feature_set["indexer"].keys():
            return feature_set
    num_triplets = feature_set["metadata"]["num_ijk"]
    num_atoms = feature_set["metadata"]["num_i"]
    num_frame_pairs = feature_set["metadata"]["num_ijkl"]
    if max_n_frames is None:
        max_n_frames = num_triplets
    else:
        max_n_frames = max(min(num_triplets, max_n_frames), 1)
    key_frame_idx = np.random.choice(num_triplets, size=max_n_frames, replace=False)
    key_atom_idx = feature_set["indexer"]["gather_idx_ijk_j"][key_frame_idx]
    num_key_frames = key_frame_idx.shape[0]
    # scatter_idx_u0ijk_Uijk = (
    #     feature_set["indexer"]["gather_idx_u0ijk_ijk"]
    #     + feature_set["indexer"]["gather_idx_u0ijk_u0"] * num_triplets
    # )
    gather_idx_UI_Uijk = (
        np.broadcast_to(
            np.arange(num_key_frames)[:, None],
            (num_key_frames, num_key_frames),
        ).flatten()
        * num_triplets
        + np.broadcast_to(
            key_frame_idx[None, :],
            (num_key_frames, num_key_frames),
        ).flatten()
    )
    gather_idx_Uijkl_Uijk = (
        np.broadcast_to(
            np.arange(num_key_frames)[:, None], (num_key_frames, num_frame_pairs)
        ).flatten()
        * num_triplets
        + np.broadcast_to(
            feature_set["indexer"]["gather_idx_ijkl_ijk"][None, :],
            (num_key_frames, num_frame_pairs),
        ).flatten()
    )
    gather_idx_Uijkl_ujkl = (
        np.broadcast_to(
            np.arange(num_key_frames)[:, None], (num_key_frames, num_frame_pairs)
        ).flatten()
        * num_triplets
        + np.broadcast_to(
            feature_set["indexer"]["gather_idx_ijkl_jkl"][None, :],
            (num_key_frames, num_frame_pairs),
        ).flatten()
    )
    gather_idx_Uijkl_ijkl = np.broadcast_to(
        np.arange(num_frame_pairs)[None, :], (num_key_frames, num_frame_pairs)
    ).flatten()

    adjacency_mat = np.zeros((num_atoms, num_atoms), dtype=np.int_)
    adjacency_mat[
        feature_set["indexer"]["gather_idx_ij_i"],
        feature_set["indexer"]["gather_idx_ij_j"],
    ] = 1
    sum_pair_path_dist = [np.eye(num_atoms, dtype=np.int_)]
    for path_length in range(3):
        sum_pair_path_dist.append(np.matmul(sum_pair_path_dist[-1], adjacency_mat))
    sum_pair_path_dist = np.stack(sum_pair_path_dist, axis=2)
    atom_pair_feature_mat = np.zeros((num_atoms, num_atoms, 4), dtype=np.float_)
    atom_pair_feature_mat[
        feature_set["indexer"]["gather_idx_ij_i"],
        feature_set["indexer"]["gather_idx_ij_j"],
    ] = feature_set["features"]["bond_encodings"]
    atom_pair_feature_mat = np.concatenate(
        [atom_pair_feature_mat, (sum_pair_path_dist > 0).astype(np.float_)], axis=2
    )
    uv_adj_mat = np.sum(sum_pair_path_dist, axis=2) > 0
    gather_idx_uv_u = np.broadcast_to(
        np.arange(num_atoms)[:, None], (num_atoms, num_atoms)
    )[uv_adj_mat]
    gather_idx_uv_v = np.broadcast_to(
        np.arange(num_atoms)[None, :], (num_atoms, num_atoms)
    )[uv_adj_mat]

    atom_frame_pair_feat_initial_ = np.concatenate(
        [
            atom_pair_feature_mat[key_atom_idx, :][
                :, feature_set["indexer"]["gather_idx_ijk_i"]
            ],
            atom_pair_feature_mat[key_atom_idx, :][
                :, feature_set["indexer"]["gather_idx_ijk_j"]
            ],
            atom_pair_feature_mat[key_atom_idx, :][
                :, feature_set["indexer"]["gather_idx_ijk_k"]
            ],
        ],
        axis=2,
    ).reshape((num_key_frames * num_triplets, atom_pair_feature_mat.shape[2] * 3))

    # Generate on-the-fly to reduce disk usage
    feature_set["indexer"].update(
        {
            "gather_idx_U_u": key_atom_idx,
            "gather_idx_I_ijk": key_frame_idx,
            "gather_idx_I_molid": np.zeros((num_key_frames,), dtype=np.int_),
            "gather_idx_UI_Uijk": gather_idx_UI_Uijk,
            "gather_idx_UI_u": np.broadcast_to(
                key_atom_idx[:, None], (num_key_frames, num_key_frames)
            ).flatten(),
            "gather_idx_UI_U": np.broadcast_to(
                np.arange(num_key_frames)[:, None], (num_key_frames, num_key_frames)
            ).flatten(),
            "gather_idx_UI_I": np.broadcast_to(
                np.arange(num_key_frames)[None, :], (num_key_frames, num_key_frames)
            ).flatten(),
            # "scatter_idx_u0ijk_Uijk": scatter_idx_u0ijk_Uijk,
            "gather_idx_Uijk_u": np.broadcast_to(
                key_atom_idx[:, None], (num_key_frames, num_triplets)
            ).flatten(),
            "gather_idx_Uijk_ijk": np.broadcast_to(
                np.arange(num_triplets)[None, :], (num_key_frames, num_triplets)
            ).flatten(),
            "gather_idx_Uijkl_Uijk": gather_idx_Uijkl_Uijk,
            "gather_idx_Uijkl_ujkl": gather_idx_Uijkl_ujkl,
            "gather_idx_Uijkl_ijkl": gather_idx_Uijkl_ijkl,
            "gather_idx_uv_u": gather_idx_uv_u,
            "gather_idx_uv_v": gather_idx_uv_v,
        }
    )
    feature_set["features"].update(
        {
            "atom_pair_encodings": atom_pair_feature_mat[uv_adj_mat],
            "atom_frame_pair_encodings": atom_frame_pair_feat_initial_,
        }
    )
    feature_set["metadata"]["num_v"] = num_atoms
    feature_set["metadata"]["num_I"] = num_key_frames
    feature_set["metadata"]["num_U"] = num_key_frames
    feature_set["metadata"]["num_Uijk"] = num_triplets * num_key_frames
    feature_set["metadata"]["num_ujkl"] = num_triplets * num_key_frames
    feature_set["metadata"]["num_Uijkl"] = num_frame_pairs * num_key_frames
    feature_set["metadata"]["num_uv"] = gather_idx_uv_u.shape[0]
    feature_set["metadata"]["num_UI"] = num_key_frames * num_key_frames
    return feature_set


def smiles2inputs(tokenizer, smiles, pad_length=128):
    """Adapted from megamolbart"""

    assert isinstance(smiles, str)
    if pad_length:
        assert pad_length >= len(smiles) + 2

    tokens = tokenizer.tokenize([smiles], pad=True)

    # Append to tokens and mask if appropriate
    if pad_length:
        for i in range(len(tokens["original_tokens"])):
            num_i = len(tokens["original_tokens"][i])
            n_pad = pad_length - len(tokens["original_tokens"][i])
            tokens["original_tokens"][i] += [tokenizer.pad_token] * n_pad
            tokens["masked_pad_masks"][i] += [1] * n_pad

    token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens["original_tokens"]))
    pad_mask = torch.tensor(tokens["masked_pad_masks"]).bool()
    encode_input = {
        "features": {"encoder_input": token_ids, "encoder_pad_mask": pad_mask},
        "indexer": {"gather_idx_i_molid": np.zeros((num_i,), dtype=np.int_)},
        "metadata": {"num_i": num_i, "num_molid": 1},
    }

    return encode_input


def _process_molecule(
    mol: Chem.Mol,
    max_path_length=-1,
    only_2d=False,
    return_mol=False,
    ref_conf_xyz=None,
):
    # All bonds are directional
    atom_encodings_list = [get_atom_encoding(atom) for atom in mol.GetAtoms()]
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    nx_graph = mol_to_graph(mol)
    bond_atom_ids = np.array(nx_graph.edges())
    if mol.GetNumBonds() == 0:
        bond_encodings_list = [np.zeros((4,)) for eid in range(len(bond_atom_ids))]
    else:
        bond_encodings_list = [
            get_bond_encoding(mol.GetBondWithIdx(eid // 2))
            for eid in range(len(bond_atom_ids))
        ]
    triangle_aaabb_ids, bab_dict, aaa_dict = compute_bond_pair_triangles(nx_graph)
    atom_idx_on_triangles = triangle_aaabb_ids[:, :3]
    # triangle_atoms_list, pos_encodings_list = compute_all_atom_positional_encodings(
    #     nx_graph, atom_idx_on_triangles
    # )
    triangle_pairs_list, stereo_encodings_list = compute_all_stereo_chemistry_encodings(
        mol,
        nx_graph,
        atom_idx_on_triangles,
        aaa_dict,
        only_2d=only_2d,
        ref_conf_xyz=ref_conf_xyz,
    )
    metadata = {
        "num_molid": 1,
        "num_ligand_atom": len(atom_encodings_list),
        "num_i": len(atom_encodings_list),
        "num_j": len(atom_encodings_list),
        "num_k": len(atom_encodings_list),
        # "num_u0": len(atom_encodings_list),
        "num_u": len(atom_encodings_list),
        "num_ij": len(bond_atom_ids),
        "num_jk": len(bond_atom_ids),
        "num_ijk": len(triangle_aaabb_ids),
        "num_jkl": len(triangle_aaabb_ids),
        "num_ligand_clique": len(triangle_aaabb_ids),
        "num_ijkl": len(triangle_pairs_list),
        # "num_u0ijk": len(triangle_atoms_list),
    }
    indexer = tensorize_indexers(
        bond_atom_ids, triangle_aaabb_ids, triangle_pairs_list  # , triangle_atoms_list,
    )
    # The target molid is alway 0
    indexer["gather_idx_i_molid"] = np.zeros((metadata["num_i"],), dtype=np.int_)
    indexer["gather_idx_ijk_molid"] = np.zeros((metadata["num_ijk"],), dtype=np.int_)
    if len(stereo_encodings_list) == 0:
        stereo_encodings = np.zeros((0, 14), dtype=np.int_)
    else:
        stereo_encodings = np.stack(stereo_encodings_list, axis=0)
    features = {
        "atomic_numbers": np.array(atomic_numbers, dtype=int),
        "atom_encodings": np.stack(atom_encodings_list, axis=0),
        "bond_encodings": np.stack(bond_encodings_list, axis=0),
        # "atom_positional_encodings": np.stack(pos_encodings_list, axis=0),
        "stereo_chemistry_encodings": stereo_encodings,
    }
    if return_mol:
        return mol, metadata, indexer, features
    return metadata, indexer, features


def process_pdb(pdb_string: str = None, load_msgpack: str = None, **kwargs):
    if load_msgpack:
        with open(load_msgpack, "rb") as data_file:
            byte_data = data_file.read()
            data_loaded = msgpack.unpackb(byte_data)
            af_protein = Protein(
                letter_sequences=data_loaded["letter_sequences"],
                atom_positions=data_loaded["atom_positions"],
                atom_mask=data_loaded["atom_mask"],
                aatype=data_loaded["aatype"],
                atomtypes=data_loaded["atomtypes"],
                residue_index=data_loaded["residue_index"],
                chain_index=data_loaded["chain_index"],
                b_factors=data_loaded["b_factors"],
            )
    else:
        af_protein = af_protein_from_pdb_string(pdb_string, **kwargs)
    return _process_protein(af_protein, **kwargs)


def _get_protein_indexer(rec_features, edge_cutoff=50):
    # Using a large cutoff here; dynamically remove edges along diffusion
    res_xyzs = rec_features["res_atom_positions"]
    n_res = len(res_xyzs)
    res_atom_masks = rec_features["res_atom_mask"]
    ca_xyzs = res_xyzs[:, 1, :]
    distances = np.linalg.norm(
        ca_xyzs[:, np.newaxis, :] - ca_xyzs[np.newaxis, :, :], axis=2
    )
    edge_mask = distances < edge_cutoff
    # Mask out residues where the backbone is not resolved
    res_mask = np.all(~res_atom_masks[:, :3], axis=1)
    edge_mask[res_mask, :] = 0
    edge_mask[:, res_mask] = 0
    res_ids = np.broadcast_to(np.arange(n_res), (n_res, n_res))
    src_nid, dst_nid = res_ids[edge_mask], res_ids.T[edge_mask]

    indexer = {
        "gather_idx_a_chainid": rec_features["res_chain_id"],
        "gather_idx_a_structid": np.zeros((n_res,), dtype=np.int_),
        "gather_idx_ab_a": src_nid,
        "gather_idx_ab_b": dst_nid,
    }
    return indexer


def _process_protein(
    af_protein: Protein,
    bounding_box=None,
    no_indexer=True,
    sample_name="",
    plddt=None,
    **kwargs,
):
    if bounding_box:
        raise NotImplementedError
        ca_pos = af_protein.atom_positions[:, 1]
        ca_in_box = np.all(
            (ca_pos > bounding_box[0]) & (ca_pos < bounding_box[1]),
            axis=1,
        )
        af_protein.atom_positions = af_protein.atom_positions[ca_in_box]
        af_protein.aatype = af_protein.aatype[ca_in_box]
        af_protein.atomtypes = af_protein.atomtypes[ca_in_box]
        af_protein.atom_mask = af_protein.atom_mask[ca_in_box]
        af_protein.chain_index = af_protein.chain_index[ca_in_box]
        af_protein.b_factors = af_protein.b_factors[ca_in_box]
    chain_seqs = [
        (sample_name + seq_data[0], seq_data[1])
        for seq_data in af_protein.letter_sequences
    ]
    chain_masks = [seq_data[2] for seq_data in af_protein.letter_sequences]
    features = {
        "res_atom_positions": af_protein.atom_positions,
        "res_type": np.int_(af_protein.aatype),
        "res_atom_types": np.int_(af_protein.atomtypes),
        "res_atom_mask": np.bool_(af_protein.atom_mask),
        "res_chain_id": np.int_(af_protein.chain_index),
        "residue_index": np.int_(af_protein.residue_index),
        "sequence_res_mask": np.bool_(np.concatenate(chain_masks)),
    }
    if plddt:
        features.update({"pLDDT": np.array(plddt) / 100})
    n_res = len(af_protein.atom_positions)
    metadata = {
        "num_structid": 1,
        "num_a": n_res,
        "num_b": n_res,
        "num_chainid": max(af_protein.chain_index) + 1,
    }
    if no_indexer:
        return {
            "metadata": metadata,
            "indexer": {
                "gather_idx_a_chainid": features["res_chain_id"],
                "gather_idx_a_structid": np.zeros((n_res,), dtype=np.int_),
            },
            "features": features,
            "misc": {"sequence_data": chain_seqs},
        }
    return {
        "metadata": metadata,
        "indexer": _get_protein_indexer(features),
        "features": features,
        "misc": {"sequence_data": chain_seqs},
    }


def crop_protein_features(sample, crop_size: int):
    original_size = sample["metadata"]["num_a"]
    if crop_size is None or crop_size >= original_size:
        return sample
    # Only supported for single-chain gapless pdbs
    if len(sample["misc"]["sequence_data"]) > 1:
        raise NotImplementedError
    if len(sample["misc"]["sequence_data"][0][1]) != original_size:
        raise NotImplementedError
    sample["metadata"]["num_a"] = crop_size
    sample["metadata"]["num_b"] = crop_size
    start_res = random.randint(0, original_size - crop_size - 1)
    for k, v in sample["features"].items():
        sample["features"][k] = v[start_res : start_res + crop_size]
    for k, v in sample["indexer"].items():
        sample["indexer"][k] = v[start_res : start_res + crop_size]
    sample["misc"]["sequence_data"][0][1] = sample["misc"]["sequence_data"][0][1][
        start_res : start_res + crop_size
    ]
    return sample


def process_template_protein_features(sample, template_sample, auto_align=True):
    # if len(sample["misc"]["sequence_data"]) > 1:
    #     warnings.warn(
    #         f"Reference structure contains multiple chains. Sequences: {sample['misc']['sequence_data']}"
    #     )
    sample_sequence = "".join(
        [seq_dat[1] for seq_dat in sample["misc"]["sequence_data"]]
    )
    sample_sequence = np.array(list(sample_sequence))
    # if len(template_sample["misc"]["sequence_data"]) > 1:
    #     warnings.warn(
    #         f"Template structure contains multiple chains. Sequences: {template_sample['misc']['sequence_data']}"
    #     )
    template_sequence = "".join(
        [seq_dat[1] for seq_dat in template_sample["misc"]["sequence_data"]]
    )
    template_sequence = np.array(list(template_sequence))
    # Brute force sequence alignment (gapless sequences only)
    assert len(sample_sequence) == len(sample["features"]["sequence_res_mask"])
    assert len(template_sequence) == len(
        template_sample["features"]["sequence_res_mask"]
    )
    if auto_align:
        alignment_scores = []
        for seq_diff in range(-len(template_sequence) + 1, len(sample_sequence)):
            alignment_scores.append(
                np.sum(
                    sample_sequence[
                        max(0, seq_diff) : min(
                            len(sample_sequence), seq_diff + len(template_sequence)
                        )
                    ]
                    == template_sequence[
                        max(0, -seq_diff) : min(
                            len(template_sequence), -seq_diff + len(sample_sequence)
                        )
                    ]
                )
            )
        opt_seq_diff = np.argmax(alignment_scores) - len(template_sequence) + 1
    else:
        opt_seq_diff = 0
    # Pad/crop the coordinate sets
    template_coords, template_atom37_mask, aligned_mask = [], [], []
    if opt_seq_diff > 0:
        aligned_mask.append(np.zeros((opt_seq_diff), dtype=np.bool_))
        template_coords.append(np.zeros((opt_seq_diff, 37, 3)))
        template_atom37_mask.append(np.zeros((opt_seq_diff, 37), dtype=np.bool_))
    template_coords.append(
        template_sample["features"]["res_atom_positions"][
            max(0, -opt_seq_diff) : min(
                len(template_sequence), -opt_seq_diff + len(sample_sequence)
            )
        ]
    )
    template_atom37_mask.append(
        template_sample["features"]["res_atom_mask"][
            max(0, -opt_seq_diff) : min(
                len(template_sequence), -opt_seq_diff + len(sample_sequence)
            )
        ]
    )
    aligned_mask.append(
        sample_sequence[
            max(0, opt_seq_diff) : min(
                len(sample_sequence), opt_seq_diff + len(template_sequence)
            )
        ]
        == template_sequence[
            max(0, -opt_seq_diff) : min(
                len(template_sequence), -opt_seq_diff + len(sample_sequence)
            )
        ]
    )
    if len(template_sequence) + opt_seq_diff < len(sample_sequence):
        end_padding = len(sample_sequence) - opt_seq_diff - len(template_sequence)
        aligned_mask.append(np.zeros((end_padding), dtype=np.bool_))
        template_coords.append(np.zeros((end_padding, 37, 3)))
        template_atom37_mask.append(np.zeros((end_padding, 37), dtype=np.bool_))
    sample["features"]["template_atom_positions"] = np.concatenate(
        template_coords, axis=0
    )
    sample["features"]["template_atom37_mask"] = np.concatenate(
        template_atom37_mask, axis=0
    )
    sample["features"]["template_alignment_mask"] = np.concatenate(aligned_mask, axis=0)
    if "pLDDT" in template_sample["features"].keys():
        template_plddt = []
        if opt_seq_diff > 0:
            template_plddt.append(np.zeros((opt_seq_diff)))
        template_plddt.append(
            template_sample["features"]["pLDDT"][
                max(0, -opt_seq_diff) : min(
                    len(template_sequence), -opt_seq_diff + len(sample_sequence)
                )
            ]
        )
        if len(template_sequence) + opt_seq_diff < len(sample_sequence):
            end_padding = len(sample_sequence) - opt_seq_diff - len(template_sequence)
            template_plddt.append(np.zeros((end_padding)))
        sample["features"]["template_pLDDT"] = np.concatenate(template_plddt, axis=0)
    #     print(sample["features"]["template_atom37_mask"].shape, sample["features"]["template_atom_positions"].shape, sample["features"]["template_pLDDT"].shape)
    return sample


def _get_rec_chain(rec_sample, chain_id: int):
    chain_mask = rec_sample["features"]["res_chain_id"] == chain_id
    features = {
        "res_atom_positions": rec_sample["features"]["res_atom_positions"][chain_mask],
        "res_type": rec_sample["features"]["res_type"][chain_mask],
        "res_atom_mask": rec_sample["features"]["res_atom_mask"][chain_mask],
        "res_chain_id": rec_sample["features"]["res_chain_id"][chain_mask],
    }
    n_res = len(features["res_atom_positions"])
    metadata = {
        "num_structid": 1,
        "num_protein_residues": n_res,
        "num_a": n_res,
        "num_b": n_res,
        "num_chainid": 1,
    }
    return {
        "metadata": metadata,
        "indexer": _get_protein_indexer(features),
        "features": features,
    }


def get_nearest_chain_id(lig_sample, rec_sample):
    # Get the chain nearest to the ligand by the minimum pairwise distance
    intermol_distmat = np.linalg.norm(
        lig_sample["features"]["sdf_coordinates"][np.newaxis, :]
        - rec_sample["features"]["res_atom_positions"][:, 1, np.newaxis, :],
        axis=2,
    )
    chain_min_dist = []
    for chain_id in range(rec_sample["metadata"]["num_chainid"]):
        chain_mask = rec_sample["features"]["res_chain_id"] == chain_id
        chain_min_dist.append(np.min(intermol_distmat[chain_mask]))
    return np.argmin(chain_min_dist)


def _lig_is_in_convex_hull(
    lig_sample, rec_sample, distance_cutoff=5.0, natm_cutoff=1024
):
    if lig_sample["metadata"]["num_i"] > natm_cutoff:
        return False
    intermol_distmat = np.linalg.norm(
        lig_sample["features"]["sdf_coordinates"][None, :]
        - rec_sample["features"]["res_atom_positions"][
            rec_sample["features"]["res_atom_mask"]
        ][:, None, :],
        axis=2,
    )
    # # Check whether 50% of heavy atoms are within 8.0A of the protein
    # return np.mean(np.min(intermol_distmat, axis=0) < distance_cutoff) > 0.5
    # Check if 10% of heavy atoms are within 5.0A of the protein
    return np.mean(np.min(intermol_distmat, axis=0) < distance_cutoff) > 0.1


def merge_protein_and_ligands(
    lig_samples,
    rec_sample,
    n_lig_patches,
    label=None,
    nearest_chain_only=False,
    random_lig_placement=False,
    filter_ligands=False,
    subsample_frames=False,
):
    if filter_ligands:
        lig_samples = [
            lig_sample
            for lig_sample in lig_samples
            if _lig_is_in_convex_hull(lig_sample, rec_sample)
        ]
    # Assign frame sampling rate to each ligand
    num_ligands = len(lig_samples)
    if num_ligands > 0:
        num_frames_sqrt = np.sqrt(
            np.array([lig_sample["metadata"]["num_ijk"] for lig_sample in lig_samples])
        )
        if (n_lig_patches > sum(num_frames_sqrt)) and subsample_frames:
            n_lig_patches = random.randint(int(sum(num_frames_sqrt)), n_lig_patches)
        max_n_frames_arr = num_frames_sqrt * (n_lig_patches / sum(num_frames_sqrt))
        max_n_frames_arr = max_n_frames_arr.astype(np.int_)
        lig_samples = [
            _attach_pair_idx_and_encodings(
                lig_sample, max_n_frames=max_n_frames_arr[lig_idx]
            )
            for lig_idx, lig_sample in enumerate(lig_samples)
        ]

    if random_lig_placement:
        # Data augmentation, randomly placing into box
        rec_coords = rec_sample["features"]["res_atom_positions"][
            rec_sample["features"]["res_atom_mask"]
        ]
        box_lbound, box_ubound = (
            np.amin(rec_coords, axis=0),
            np.amax(rec_coords, axis=0),
        )
        for sid, lig_sample in enumerate(lig_samples):
            lig_coords = lig_sample["features"]["sdf_coordinates"]
            lig_center = np.mean(lig_coords, axis=0)
            is_clash = True
            padding = 0
            while is_clash:
                padding += 1.0
                new_center = np.random.uniform(
                    low=box_lbound - padding, high=box_ubound + padding
                )
                new_lig_coords = lig_coords + (new_center - lig_center)[None, :]
                intermol_distmat = np.linalg.norm(
                    new_lig_coords[None, :] - rec_coords[:, None, :],
                    axis=2,
                )
                if np.amin(intermol_distmat) > 4.0:
                    is_clash = False
            lig_samples[sid]["features"]["augmented_coordinates"] = new_lig_coords
            del lig_samples[sid]["features"]["sdf_coordinates"]
    lig_sample_merged = collate_numpy(lig_samples)
    if nearest_chain_only:
        nearest_chain = get_nearest_chain_id(lig_sample_merged, rec_sample)
        rec_sample = _get_rec_chain(rec_sample, nearest_chain)
        return merge_protein_and_ligands(
            lig_sample_merged,
            rec_sample,
            label=label,
            nearest_chain_only=False,
        )
    merged = {
        "metadata": {**lig_sample_merged["metadata"], **rec_sample["metadata"]},
        "features": {**lig_sample_merged["features"], **rec_sample["features"]},
        "indexer": {**lig_sample_merged["indexer"], **rec_sample["indexer"]},
        "misc": {**lig_sample_merged["misc"], **rec_sample["misc"]},
    }
    merged["metadata"]["num_structid"] = 1
    if "num_molid" in merged["metadata"]:
        merged["indexer"]["gather_idx_i_structid"] = np.zeros(
            lig_sample_merged["metadata"]["num_i"], dtype=np.int_
        )
        merged["indexer"]["gather_idx_ijk_structid"] = np.zeros(
            lig_sample_merged["metadata"]["num_ijk"], dtype=np.int_
        )
    assert np.sum(merged["features"]["res_atom_mask"]) > 0
    if label is not None:
        merged["labels"] = np.array([label])
    return merged


def to_torch(sample):
    if sample is None:
        return None
    new_sample = {}
    new_sample["metadata"] = sample["metadata"]
    new_sample["features"] = {
        k: torch.FloatTensor(v.copy()) for k, v in sample["features"].items()
    }
    new_sample["indexer"] = {
        k: torch.LongTensor(v.copy()) for k, v in sample["indexer"].items()
    }
    if "labels" in sample.keys():
        new_sample["labels"] = {
            k: torch.FloatTensor(v.copy()) for k, v in sample["labels"].items()
        }
    if "misc" in sample.keys():
        new_sample["misc"] = sample["misc"]
    return new_sample


@deprecated
def generate_patches(sample, n_rec_patch=32, max_n_lig_patch=16):
    # Segment the receptor chains
    n_chains = sample["metadata"]["num_chain"]
    assert n_chains < n_rec_patch
    assert sample["metadata"]["num_a"] > n_rec_patch
    soft_pid = np.arange(sample["metadata"]["num_a"], dtype=np.float_) / n_rec_patch
    patch_counter = 0
    patch_idx, pos_in_patch = [], []
    for chain_id in range(n_chains):
        chain_mask = sample["features"]["res_chain_id"] == chain_id
        chain_length = np.sum(chain_mask)
        n_patch_in_chain = int(max(soft_pid[chain_mask])) + 1 - patch_counter
        patch_size = chain_length / n_rec_patch
        sub_patch_idx, sub_pos_in_patch = np.divmod(np.arange(chain_length), patch_size)
        patch_idx.append(patch_counter + sub_patch_idx.astype(np.int_))
        pos_in_patch.append(pos_in_patch.append(sub_pos_in_patch))
        patch_counter += n_patch_in_chain
    res_patch_idx = np.concatenate(patch_idx, axis=0)
    sample["indexer"]["gather_idx_a_A"] = res_patch_idx
    sample["features"]["position_in_patch"] = np.concatenate(pos_in_patch, axis=0)
    patch_sizes = [
        np.count_nonzero(sample["indexer"]["gather_idx_a_A"] == A)
        for A in range(n_rec_patch)
    ]
    sample["features"]["target_patch_size"] = np.array(patch_sizes)[res_patch_idx]
    # 08/04: dynamically resampling small-molecule patches within the model
    # # Merged ligand graph clustering
    # bond_weight = np.sum(
    #     np.arange(1, 5)[np.newaxis, :] * sample["features"]["bond_encodings"], axis=-1
    # )
    # adj_mat = scipy.sparse.coo_matrix(
    #     bond_weight,
    #     (sample["indexer"]["gather_idx_ij_i"], sample["indexer"]["gather_idx_ij_j"]),
    # )
    n_lig_patch = min(sample["metadata"]["num_j"], max_n_lig_patch)
    # clustering = AgglomerativeClustering(n_lig_patch, affinity="precomputed").fit(
    #     1 / adj_mat
    # )
    # sample["indexer"]["gather_idx_i_I"] = clustering.labels_ + n_rec_patch
    # sample["indexer"]["gather_idx_i_Ilig"] = clustering.labels_
    sample["feature"]["patch_mask"] = (
        np.arange(n_rec_patch + max_n_lig_patch) < n_rec_patch + n_lig_patch
    )
    # Note: IJ-indexers are constructed on the fly
    sample["metadata"]["num_patch"] = n_rec_patch + max_n_lig_patch
    sample["metadata"]["num_A"] = n_rec_patch
    sample["metadata"]["num_I"] = n_lig_patch
    return sample


def inplace_to_torch(sample):
    if sample is None:
        return None
    sample["features"] = {
        k: torch.FloatTensor(v) for k, v in sample["features"].items()
    }
    sample["indexer"] = {k: torch.LongTensor(v) for k, v in sample["indexer"].items()}
    if "labels" in sample.keys():
        sample["labels"] = {
            k: torch.FloatTensor(v) for k, v in sample["labels"].items()
        }
    return sample


def to_torch(sample):
    if sample is None:
        return None
    new_sample = {}
    new_sample["features"] = {
        k: torch.FloatTensor(v) for k, v in sample["features"].items()
    }
    new_sample["indexer"] = {
        k: torch.LongTensor(v) for k, v in sample["indexer"].items()
    }
    if "labels" in sample.keys():
        new_sample["labels"] = {
            k: torch.FloatTensor(v) for k, v in sample["labels"].items()
        }
    new_sample["metadata"] = sample["metadata"]
    return new_sample


def inplace_to_cuda(sample):
    sample["features"] = {k: v.cuda() for k, v in sample["features"].items()}
    sample["indexer"] = {k: v.cuda() for k, v in sample["indexer"].items()}
    if "labels" in sample.keys():
        sample["labels"] = sample["labels"].cuda()
    return sample


def inplace_to_device(sample, device):
    sample["features"] = {k: v.to(device) for k, v in sample["features"].items()}
    sample["indexer"] = {k: v.to(device) for k, v in sample["indexer"].items()}
    if "labels" in sample.keys():
        sample["labels"] = sample["labels"].to(device)
    return sample


def featurize_protein_and_ligands(
    lig_paths,
    rec_path,
    n_lig_patches,
    chain_id=None,
    template_path=None,
    enforce_sanitization=False,
    discard_sdf_coords=False,
    **kwargs,
):
    assert rec_path is not None
    if lig_paths is None:
        lig_paths = []
    if isinstance(lig_paths, str):
        lig_paths = [lig_paths]
    out_mol = None
    lig_samples = []
    for lig_path in lig_paths:
        try:
            lig_sample, mol_ref = process_mol_file(
                lig_path,
                sanitize=True,
                return_mol=True,
                discard_coords=discard_sdf_coords,
            )
        except Exception as e:
            if enforce_sanitization:
                raise
            warnings.warn(
                f"RDKit sanitization failed for ligand {lig_path}: {e}, loading raw attributes"
            )
            lig_sample, mol_ref = process_mol_file(
                lig_path,
                sanitize=False,
                return_mol=True,
                discard_coords=discard_sdf_coords,
            )
        lig_samples.append(lig_sample)
        if out_mol is None:
            out_mol = mol_ref
        else:
            out_mol = AllChem.CombineMols(out_mol, mol_ref)
    rec_sample = process_pdb(open(rec_path).read(), chain_id=chain_id, **kwargs)
    merged_sample = merge_protein_and_ligands(
        lig_samples,
        rec_sample,
        n_lig_patches=n_lig_patches,
        label=None,
        filter_ligands=False,
    )
    if template_path is not None:
        if template_path == "self":
            template_path = rec_path
        template_sample = process_pdb(open(template_path).read(), **kwargs)
        merged_sample = process_template_protein_features(
            merged_sample, template_sample
        )
    return merged_sample, out_mol


def write_conformer_sdf(mol, confs: np.array = None, out_path="test_results/debug.sdf"):
    if confs is None:
        w = Chem.SDWriter(out_path)
        w.write(mol)
        w.close()
        return 0
    mol.RemoveAllConformers()
    for i in range(len(confs)):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for j in range(mol.GetNumAtoms()):
            x, y, z = confs[i, j].tolist()
            conf.SetAtomPosition(j, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

    w = Chem.SDWriter(out_path)
    try:
        for cid in range(len(confs)):
            w.write(mol, confId=cid)
    except:
        w.SetKekulize(False)
        for cid in range(len(confs)):
            w.write(mol, confId=cid)
    w.close()
    return 0


def write_pdb_single(result, out_path="test_results/debug.pdb", model=1, b_factors=None):
    protein = from_prediction(result["features"], result, b_factors=b_factors)
    out_string = to_pdb(protein, model=model)
    with open(out_path, "w") as of:
        of.write(out_string)


def write_pdb_models(results, out_path="test_results/debug.pdb", b_factors=None):
    with open(out_path, "w") as of:
        for mid, result in enumerate(results):
            protein = from_prediction(result["features"], result, b_factors=b_factors[mid] if b_factors is not None else None)
            out_string = to_pdb(protein, model=mid + 1)
            of.write(out_string)
        of.write("END")


if __name__ == "__main__":
    import pprint
    import sys

    import tqdm

    pp = pprint.PrettyPrinter(indent=4)
    # Testing smiles featurization
    test_smiles = sys.argv[1]
    test_metadata, test_indexer, test_features = process_smiles(
        test_smiles, max_path_length=4
    )
    pp.pprint(test_metadata)
    pp.pprint(test_indexer)
    pp.pprint(test_features)

    # Testing reproducibility
    for _ in tqdm.tqdm(range(100)):
        rep_metadata, rep_indexer, rep_features = process_smiles(
            test_smiles, max_path_length=4
        )
        for k in test_metadata.keys():
            assert test_metadata[k] == rep_metadata[k]
        for k in test_features.keys():
            try:
                assert np.all(test_features[k] == rep_features[k])
            except AssertionError:
                print(k, test_features[k] - rep_features[k])
        for k in test_indexer.keys():
            assert np.all(test_indexer[k] == rep_indexer[k])
