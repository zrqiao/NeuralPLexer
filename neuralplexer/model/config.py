import os
from pathlib import Path

import ml_collections

from neuralplexer.data.pipeline import process_mol_file, process_pdb


def get_base_config():
    return DEFAULT_CONFIG


def attach_task_config(config, task, **kwargs):
    if task == "graph_regression":
        config = _attach_regression_task_config(config, **kwargs)
    elif task == "conformer_pretraining":
        config = _attach_ligand_pretraining_task_config(config)
    elif task == "protein_pretraining":
        config = _attach_protein_denoising_task_config(config)
        config.task.task_type = task
    elif task in [
        "LBA",
        "all_atom_prediction",
    ]:
        config = _attach_binding_task_config(config)
        config.task.task_type = task
    else:
        raise NotImplementedError
    return config


def _attach_regression_task_config(config, dataset=None):
    # Deprecated, for debugging purposes
    def get_ds_metadata(ds_name):
        if ds_name == "esol":
            return "delaney-processed", "logsolv"
        elif ds_name == "lipo":
            return "Lipophilicity", "exp"
        elif ds_name == "freesolv":
            return "SAMPL", "expt"
        else:
            raise NotImplementedError

    csv_name, label_name = get_ds_metadata(dataset)
    config.task = ml_collections.ConfigDict(
        {
            "task_type": "graph_regression",
            "batch_size": 8,
            "init_learning_rate": 5e-4,
            "max_epoch": 500,
            "label_name": label_name,
            "dataset_path": f"datasets/{csv_name}.csv",
            "split_ratio": [0.8, 0.1, 0.1],
            "only_2d": True,
        }
    )
    return config


def _attach_ligand_pretraining_task_config(config):
    config.task = ml_collections.ConfigDict(
        {
            "task_type": "geometry_pretraining",
            "max_iter_per_epoch": 5000,
            "batch_size": 8,
            "init_learning_rate": 3e-4,
            "max_epoch": 20,
            "n_conformers_per_mol": 1,
            "n_modes": 64,
            "dataset_path": {
                "supervised": "/data/datasets/chemicalchecker_processed_012623",
                "conformer": "/data/datasets/10kD_conformers_processed_012623",
            },
            "split_ratio": [0.98, 0.01, 0.01],
            "dropout": 0.1,
            "use_ema": False,
            "max_masking_rate": 0.5,
            "only_2d": False,
            "global_heads": 32,
            "global_head_dim": 32,
            "label_dim": 17702,
        }
    )
    return config


def _attach_protein_denoising_task_config(config):
    config.task = ml_collections.ConfigDict(
        {
            "pretrained": None,
            "ligands": False,
            "freeze_protein_encoder": False,
            "freeze_ligand_encoder": True,
            "batch_size": 2,
            "init_learning_rate": 5e-4,
            "max_epoch": 310,
            "label_name": None,
            "split_csv": "datasets/openfold/uniclust30_270k_af2_100222.csv",
            "sequence_crop_size": 400,
            "edge_crop_size": None,
            "n_modes": 8,
            "dropout": 0.1,
            "pretraining": True,
            "use_template": False,
            "use_plddt": False,
            "single_protein_batch": True,
            "global_score_loss_weight": 1.0,
            "local_distgeom_loss_weight": 0.1,
            "clash_loss_weight": 1e-4,
            "drmsd_loss_weight": 0.1,
            "distogram_loss_weight": 0.1,
            "global_max_sigma": 5.0,
            "internal_max_sigma": 2.0,
            # Sampling configs
            "frozen_backbone": False,
            "constrained_inpainting": False,
        }
    )
    return config


def _attach_binding_task_config(config):
    config.task = ml_collections.ConfigDict(
        {
            "pretrained": None,
            "ligands": True,
            "batch_size": 16,
            "epoch_frac": 1.0,
            "init_learning_rate": 2e-4,
            "max_epoch": 10,
            "label_name": None,
            "dataset_path": "/data/",
            "split_csv": "training_index/013023_general_pretraining_set_timesplit-pocketminer_reviewed.csv",
            "sequence_crop_size": 1600,
            "edge_crop_size": 80000,  # Dynamic batching max_n_edges
            "max_masking_rate": 0.0,
            "n_modes": 8,
            "dropout": 0.01,
            "use_ema": False,
            # "pretraining": True,
            "freeze_protein_encoder": False,
            "freeze_ligand_encoder": False,
            "use_template": True,
            "use_plddt": False,
            "template_key": "self",
            "block_contact_decoding_scheme": "beam",
            "frozen_backbone": False,  # deprecated
            "single_protein_batch": True,
            "contact_loss_weight": 0.2,
            "global_score_loss_weight": 0.2,
            "ligand_score_loss_weight": 0.1,
            "clash_loss_weight": 10.0,
            "local_distgeom_loss_weight": 10.0,
            "drmsd_loss_weight": 2.0,
            "distogram_loss_weight": 0.05,
            "plddt_loss_weight": 1.0,
            "global_max_sigma": 5.0,
            "internal_max_sigma": 2.0,
            "sde_decay_rate": 6.0,
            "detect_covalent": True,
            # Sampling configs
            "constrained_inpainting": True,
        }
    )
    return config


DEFAULT_CONFIG = ml_collections.ConfigDict(
    {
        "mol_encoder": {
            "node_channels": 512,
            "pair_channels": 64,
            "n_atom_encodings": 23,
            "n_bond_encodings": 4,
            "n_atom_pos_encodings": 6,
            "n_stereo_encodings": 14,
            "n_attention_heads": 8,
            "attention_head_dim": 8,
            "hidden_dim": 2048,
            "max_path_integral_length": 6,
            "n_transformer_stacks": 8,
            "n_patches": 32,
            "checkpoint_file": None,
            "from_pretrained": False,
            "megamolbart": None,
        },
        "protein_encoder": {
            "use_esm_embedding": True,
            "esm_version": "esm2_t33_650M_UR50D",
            "esm_repr_layer": 33,
            "residue_dim": 512,
            "n_aa_types": 21,
            "atom_padding_dim": 37,
            "n_atom_types": 4,  # [C, N, O, S]
            "n_patches": 96,
            "n_attention_heads": 8,
            "scalar_dim": 16,
            "point_dim": 4,
            "pair_dim": 64,
            "n_heads": 8,
            "head_dim": 8,
            "max_residue_degree": 32,
            "n_encoder_stacks": 2,
        },
        "contact_predictor": {
            "n_stacks": 4,
        },
        "score_head": {
            "fiber_dim": 64,
            "hidden_dim": 512,
            "n_stacks": 4,
            "max_atom_degree": 8,
        },
        "latent_model": "default",
    }
)


def get_standard_aa_features():
    from neuralplexer.af_common.residue_constants import restype_1to3 as af_restype_1to3
from neuralplexer.af_common.residue_constants import restypes as af_restypes

    with open(
        os.path.join(
            Path(__file__).parent.parent.absolute(),
            "data",
            "chemical",
            "20AA_template_peptide.pdb",
        ),
        "r",
    ) as standard_pdb_stream:
        standard_aa_template_featset = process_pdb(standard_pdb_stream.read())
    standard_aa_graph_featset = [
        process_mol_file(
            os.path.join(
                Path(__file__).parent.parent.absolute(),
                "data",
                "chemical",
                f"{af_restype_1to3[aa_code]}.pdb",
            ),
            sanitize=True,
            pair_feats=True,
        )
        for aa_code in af_restypes
    ]
    return standard_aa_template_featset, standard_aa_graph_featset
