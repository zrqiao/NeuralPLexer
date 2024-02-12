"""Miscellaneous tensor index trackers and batching utils"""

import numpy as np
import torch


def iterable_query(k1, k2, dc):
    return [d[k1][k2] for d in dc]


def collate_idx_tensors(idx_ten_list, dst_sample_sizes):
    device = idx_ten_list[0].device
    elewise_dst_offsets = torch.repeat_interleave(
        torch.cumsum(
            torch.tensor([0] + dst_sample_sizes[:-1], dtype=torch.long, device=device),
            dim=0,
        ),
        torch.tensor(
            [idx_ten.size(0) for idx_ten in idx_ten_list],
            dtype=torch.long,
            device=device,
        ),
    )
    col_idx_ten = torch.cat(idx_ten_list, dim=0).add_(elewise_dst_offsets)
    return col_idx_ten


def collate_idx_numpy(idx_ten_list, dst_sample_sizes):
    elewise_dst_offsets = np.repeat(
        np.cumsum(np.array([0] + dst_sample_sizes[:-1], dtype=np.int_), axis=0),
        np.array([idx_ten.shape[0] for idx_ten in idx_ten_list], dtype=np.int_),
    )
    col_idx_ten = np.concatenate(idx_ten_list, axis=0) + elewise_dst_offsets
    return col_idx_ten


def tensorize_indexers(
    bond_atom_ids,
    triangle_ids,
    triangle_pair_ids,
    prop_init_ids=None,
    prop_ids=None,
    prop_pair_ids=None,
    allow_dummy=False,
):
    # Incidence matrices flattened - querying nodes from hyperedges
    if len(bond_atom_ids) == 0:
        if not allow_dummy:
            raise ValueError("There must be at least one bond (ij), got 0")
        bond_atom_ids = np.zeros((0, 2), dtype=np.int_)
    if len(triangle_ids) == 0:
        if not allow_dummy:
            raise ValueError("There must be at least one triplet (ijk), got 0")
        triangle_ids = np.zeros((0, 5), dtype=np.int_)
        prop_init_ids = np.zeros((0, 3), dtype=np.int_)
        # prop_ids = np.zeros((0, 2), dtype=np.int_)
    if len(triangle_pair_ids) == 0:
        triangle_pair_ids = np.zeros((0, 2), dtype=np.int_)
        # prop_pair_ids = np.zeros((0, 3), dtype=np.int_)
    indexer = {
        "gather_idx_ij_i": np.array(bond_atom_ids[:, 0], dtype=np.int_),
        "gather_idx_ij_j": np.array(bond_atom_ids[:, 1], dtype=np.int_),
        "gather_idx_ijk_i": np.array(triangle_ids[:, 0], dtype=np.int_),
        "gather_idx_ijk_j": np.array(triangle_ids[:, 1], dtype=np.int_),
        "gather_idx_ijk_k": np.array(triangle_ids[:, 2], dtype=np.int_),
        "gather_idx_ijk_ij": np.array(triangle_ids[:, 3], dtype=np.int_),
        "gather_idx_ijk_jk": np.array(triangle_ids[:, 4], dtype=np.int_),
        "gather_idx_ijkl_ijk": np.array(triangle_pair_ids[:, 0], dtype=np.int_),
        "gather_idx_ijkl_jkl": np.array(triangle_pair_ids[:, 1], dtype=np.int_),
        # "gather_idx_u0ijk_ijk": np.array(prop_init_ids[:, 0], dtype=np.int_),
        # "gather_idx_u0ijk_u0": np.array(prop_init_ids[:, 1], dtype=np.int_),
    }
    return indexer


def collate_samples(list_of_samples: list, exclude=[]):
    list_of_samples = [sample for sample in list_of_samples if sample is not None]
    batch_metadata = dict()
    for key in list_of_samples[0]["metadata"].keys():
        batch_metadata[f"{key}_per_sample"] = iterable_query(
            "metadata", key, list_of_samples
        )
        if key.startswith("num_"):
            if key.endswith("_per_sample"):
                # Track the batch-level summary only
                continue
            batch_metadata[key] = sum(batch_metadata[f"{key}_per_sample"])

    batch_indexer = dict()
    for indexer_name in list_of_samples[0]["indexer"].keys():
        _, _, i1, i2 = indexer_name.split("_")
        batch_indexer[indexer_name] = collate_idx_tensors(
            iterable_query("indexer", indexer_name, list_of_samples),
            batch_metadata[f"num_{i2}_per_sample"],
        )

    batch_features = dict()
    for feature_name in list_of_samples[0]["features"].keys():
        if feature_name in exclude:
            continue
        batch_features[feature_name] = torch.cat(
            iterable_query("features", feature_name, list_of_samples), dim=0
        ).float()
    ret_batch = {
        "metadata": batch_metadata,
        "indexer": batch_indexer,
        "features": batch_features,
        "batch_size": len(list_of_samples),
    }
    if "misc" in list_of_samples[0].keys():
        ret_batch["misc"] = {
            key: [x for sample in list_of_samples for x in sample["misc"][key]]
            for key in list_of_samples[0]["misc"].keys()
        }
    if "labels" in list_of_samples[0].keys():
        ret_batch["labels"] = dict()
        for label_name in list_of_samples[0]["labels"].keys():
            ret_batch["labels"][label_name] = torch.stack(
                iterable_query("labels", label_name, list_of_samples), dim=0
            ).float()
    return ret_batch


def collate_numpy(list_of_samples: list):
    list_of_samples = [sample for sample in list_of_samples if sample is not None]
    if len(list_of_samples) == 0:
        return {
            "metadata": {},
            "indexer": {},
            "features": {},
            "labels": {},
            "misc": {},
            "batch_size": 0,
        }
    batch_metadata = dict()
    for key in list_of_samples[0]["metadata"].keys():
        batch_metadata[f"{key}_per_sample"] = iterable_query(
            "metadata", key, list_of_samples
        )
        if key.startswith("num_"):
            if key.endswith("_per_sample"):
                continue
            batch_metadata[key] = sum(batch_metadata[f"{key}_per_sample"])

    batch_indexer = dict()
    for indexer_name in list_of_samples[0]["indexer"].keys():
        _, _, i1, i2 = indexer_name.split("_")
        batch_indexer[indexer_name] = collate_idx_numpy(
            iterable_query("indexer", indexer_name, list_of_samples),
            batch_metadata[f"num_{i2}_per_sample"],
        )

    batch_features = dict()
    for feature_name in list_of_samples[0]["features"].keys():
        batch_features[feature_name] = np.concatenate(
            iterable_query("features", feature_name, list_of_samples), axis=0
        ).astype(np.float32)
    batch_misc = {
        key: [x for sample in list_of_samples for x in sample["misc"][key]]
        for key in list_of_samples[0]["misc"].keys()
    }
    if "labels" not in list_of_samples[0].keys():
        return {
            "metadata": batch_metadata,
            "indexer": batch_indexer,
            "features": batch_features,
            "misc": batch_misc,
            "batch_size": len(list_of_samples),
        }

    batch_labels = dict()
    for label_name in list_of_samples[0]["labels"].keys():
        batch_labels[label_name] = np.stack(
            iterable_query("labels", label_name, list_of_samples), axis=0
        ).astype(np.float32)
    return {
        "metadata": batch_metadata,
        "indexer": batch_indexer,
        "features": batch_features,
        "labels": batch_labels,
        "misc": batch_misc,
        "batch_size": len(list_of_samples),
    }
