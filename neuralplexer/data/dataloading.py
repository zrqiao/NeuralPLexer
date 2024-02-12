import glob
import gzip
import json
import os
import pickle
import random
from abc import ABC

import lmdb
import msgpack
import msgpack_numpy as m
import numpy as np
import pandas as pd
import torch
from deprecated import deprecated
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from neuralplexer.data.indexers import collate_samples
from neuralplexer.data.molops import get_conformers_as_tensor
from neuralplexer.data.pipeline import (_attach_pair_idx_and_encodings,
                                        _process_molecule,
                                        crop_protein_features,
                                        inplace_to_torch,
                                        merge_protein_and_ligands,
                                        process_mol_file, process_pdb,
                                        process_smiles,
                                        process_template_protein_features)

m.patch()


def load_msgpack(path):
    with open(path, "rb") as data_file:
        byte_data = data_file.read()
        sample = msgpack.unpackb(byte_data)
    return sample


class CSVDataset(Dataset, ABC):
    def __init__(self, dataset_path, label_name, max_pi_length, only_2d):
        super(CSVDataset, self).__init__()
        self.csv_path = dataset_path
        self.label_name = label_name
        self.only_2d = only_2d
        self.max_pi_length = max_pi_length
        self.df = pd.read_csv(self.csv_path)
        self._process()

    def _process(self):
        self.samples = []
        for index, row in tqdm(self.df.iterrows()):
            try:
                metadata, indexer, features = process_smiles(
                    row["smiles"],
                    max_path_length=self.max_pi_length,
                    only_2d=self.only_2d,
                )
                label = torch.Tensor([row[self.label_name]])
                self.samples.append(
                    inplace_to_torch(
                        {
                            "metadata": metadata,
                            "indexer": indexer,
                            "features": features,
                            "labels": label,
                        }
                    )
                )
            except Exception:
                print("Failed:", index, row["smiles"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def try_process_smi(max_pi_length, only_2d, n_conformers_per_mol, smiles):
    try:
        mol, metadata, indexer, features = process_smiles(
            smiles, max_path_length=max_pi_length, only_2d=only_2d, return_mol=True
        )
        new_conformer_tensor = get_conformers_as_tensor(
            mol, n_conformers_per_mol
        ).transpose(1, 0, 2)
        # Geometries in [n_atoms, n_confs, 3]
        new_sample = {
            "metadata": metadata,
            "indexer": indexer,
            "features": features,
            "labels": new_conformer_tensor,
        }
        return new_sample
    except Exception as e:
        print("Failed:", smiles, e)
        return None


def try_process_mol(max_pi_length, only_2d, mol):
    try:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
        conf_xyz = np.array(mol.GetConformer().GetPositions())
        metadata, indexer, features = _process_molecule(
            mol, max_path_length=max_pi_length, only_2d=only_2d, ref_conf_xyz=conf_xyz
        )
        conf_xyz = conf_xyz[:, np.newaxis, :]
        # Geometries in [n_atoms, 1, 3]
        new_sample = {
            "metadata": metadata,
            "indexer": indexer,
            "features": features,
            "labels": conf_xyz,
        }
        return new_sample
    except Exception as e:
        print("Failed:", e)
        return None


class MsgpackDataset(Dataset, ABC):
    def __init__(self, dataset_path):
        super(MsgpackDataset, self).__init__()
        self.keys = glob.glob(dataset_path + "/*.msgpack")
        self.weights = []
        for key in self.keys:
            if key.split("/")[-1].startswith("PubChemQC"):
                self.weights.append(0.25)
            elif key.split("/")[-1].startswith("GEOM"):
                self.weights.append(0.25)
            elif key.split("/")[-1].startswith("PEPCONF"):
                self.weights.append(5.0)
            else:
                self.weights.append(1.0)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        sample = process_mol_file(self.keys[index])
        sample = _attach_pair_idx_and_encodings(sample, max_n_frames=32)
        # sample["labels"] = sample["features"]["sdf_coordinates"][:, np.newaxis, :]
        # if sample["features"]["stereo_chemistry_encodings"].shape[1] != 14:
        #     # Manual fix for bad features
        #     return None
        return inplace_to_torch(sample)


class ConformerDataset(Dataset, ABC):
    def __init__(self, dataset_path, n_conformers_per_mol, max_pi_length, only_2d):
        super(ConformerDataset, self).__init__()
        self.csv_path = dataset_path
        self.only_2d = only_2d
        self.n_conformers_per_mol = n_conformers_per_mol
        self.max_pi_length = max_pi_length
        self.df = pd.read_csv(self.csv_path)
        self.smiles = self.df["smiles"].sample(n=128000, random_state=1).tolist()
        # Debugging, just do partial loading
        self._process()

    def _process(self):
        self.samples = []
        from functools import partial

        from tqdm.contrib.concurrent import process_map

        _p = partial(
            try_process_smi, self.max_pi_length, self.only_2d, self.n_conformers_per_mol
        )
        all_smiles = [sm for sm in self.smiles if len(sm) < 60]
        samples = process_map(_p, all_smiles, max_workers=32, chunksize=4)
        self.samples = []
        for sample in tqdm(samples):
            if sample is None:
                continue
            self.samples.append(inplace_to_torch(sample))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class GEOMDataset(Dataset, ABC):
    def __init__(self, dataset_path, n_total, max_pi_length, only_2d):
        super(GEOMDataset, self).__init__()
        self.json_path = dataset_path + "summary_drugs.json"
        with open(self.json_path, "r") as f:
            drugs_summ = json.load(f)
        all_entries = []
        for smi, v in tqdm(drugs_summ.items()):
            try:
                all_entries.append(
                    [smi, dataset_path + v["pickle_path"], v["uniqueconfs"]]
                )
            except Exception as e:
                print(f"Failed loading {smi}:", e)
        self.df = pd.DataFrame(
            data=all_entries, columns=["smiles", "pickle_path", "uniqueconfs"]
        )
        print(self.df)
        self.only_2d = only_2d
        self.n_conformers_per_mol = 1
        self.max_pi_length = max_pi_length
        self.pickle_paths = (
            self.df["pickle_path"]
            .sample(n=n_total, random_state=1, weights=self.df["uniqueconfs"])
            .tolist()
        )
        self._process()

    def _process(self):
        mols = []
        for pickle_file in tqdm(self.pickle_paths):
            with open(pickle_file, "rb") as f:
                dic = pickle.load(f)
                sample_weights = np.array(
                    [c["boltzmannweight"] for c in dic["conformers"]]
                )
                sample_weights = sample_weights / sample_weights.sum()
                conf_id = np.random.choice(len(sample_weights), p=sample_weights)
                mol = dic["conformers"][conf_id]["rd_mol"]
            if not mol:
                print("Failed to load file: ", pickle_file)
                continue
            mols.append(mol)
        self.samples = []
        from functools import partial

        from tqdm.contrib.concurrent import process_map

        _p = partial(try_process_mol, self.max_pi_length, self.only_2d)
        samples = process_map(_p, mols, max_workers=16, chunksize=4)
        self.samples = []
        for sample in tqdm(samples):
            if sample is None:
                continue
            self.samples.append(inplace_to_torch(sample))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class MolPropDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.task.batch_size
        self.split_ratio = config.task.split_ratio
        self.n_total = self.config.task.max_iter_per_epoch * self.config.task.batch_size

    def prepare_data(self):
        # Put everything here to avoid leakage
        self.train_dataset = {}
        self.val_dataset = {}
        self.test_dataset = {}
        for k, dataset_path in self.config.task.dataset_path.items():
            if self.config.task.task_type == "geometry_pretraining":
                base_dataset = MsgpackDataset(dataset_path)
            else:
                base_dataset = CSVDataset(
                    dataset_path,
                    self.config.task.label_name,
                    self.config.mol_encoder.max_path_integral_length,
                    self.config.task.only_2d,
                )
            # train/val split
            n_train = int(len(base_dataset) * self.split_ratio[0])
            n_valid = int(len(base_dataset) * self.split_ratio[1])
            n_test = int(len(base_dataset)) - n_train - n_valid
            ds_train, ds_val, ds_test = random_split(
                base_dataset,
                [
                    n_train,
                    n_valid,
                    n_test,
                ],
                generator=torch.Generator().manual_seed(42),
            )

            # assign to use in dataloaders
            self.train_dataset[k] = ds_train
            self.val_dataset[k] = ds_val
            self.test_dataset[k] = ds_test

    def train_dataloader(self) -> DataLoader:
        n_total = int(self.n_total * self.split_ratio[0])
        return CombinedLoader(
            {
                k: self._get_dataloader(ds, n_total)
                for k, ds in self.train_dataset.items()
            },
            mode="min_size",
        )

    def val_dataloader(self) -> DataLoader:
        n_total = int(self.n_total * self.split_ratio[1])
        return CombinedLoader(
            {
                k: self._get_dataloader(ds, n_total)
                for k, ds in self.val_dataset.items()
            },
            mode="min_size",
        )

    def test_dataloader(self) -> DataLoader:
        n_total = int(self.n_total * self.split_ratio[2])
        return CombinedLoader(
            {
                k: self._get_dataloader(ds, n_total)
                for k, ds in self.test_dataset.items()
            },
            mode="min_size",
        )

    def _get_dataloader(self, dataset, n_total, **kwargs):
        sample_weights = np.array(
            [dataset.dataset.weights[idx] for idx in dataset.indices]
        )
        sample_weights = sample_weights / np.sum(sample_weights)
        sampled_indices = np.random.choice(len(dataset), n_total, p=sample_weights)
        print(sampled_indices[:10])
        subdataset = torch.utils.data.Subset(dataset, sampled_indices)
        return DataLoader(
            subdataset,
            batch_size=self.batch_size,
            collate_fn=collate_samples,
            num_workers=4,
            drop_last=True,
            **kwargs,
        )


class BindingDataModule(LightningDataModule):
    def __init__(self, config, model=None):
        super().__init__()
        self.config = config
        self.batch_size = config.task.batch_size
        self.lmdb_path = config.task.lmdb_path
        self.epoch_frac = config.task.epoch_frac
        if self.config.mol_encoder.model_name == "megamolbart":
            assert model is not None
            self.tokenizer = model.ligand_encoder.tokenizer
        else:
            self.tokenizer = None
        self.datasets = {}

    def setup(self, stage):
        self.split_df = pd.read_csv(self.config.task.split_csv).replace({np.nan: None})
        subsampling = self.config.task.training_frac
        if self.config.task.use_template:
            template_key = self.config.task.template_key
        else:
            template_key = None
        if stage == "fit":
            mode = "train"
            self.train_ds = LMDBMsgpackPDBDataset(
                self.split_df[self.split_df[mode] == True],
                self.lmdb_path,
                max_n_lig_patches=self.config.mol_encoder.n_patches,
                ligands=self.config.task.ligands,
                subsampling=subsampling,
                template_key=template_key,
            )
            mode = "val"
            self.val_ds = LMDBMsgpackPDBDataset(
                self.split_df[self.split_df[mode] == True],
                self.lmdb_path,
                max_n_lig_patches=self.config.mol_encoder.n_patches,
                ligands=self.config.task.ligands,
                subsampling=subsampling,
                template_key=template_key,
            )
        elif stage == "test":
            mode = "test"
            self.test_ds = LMDBMsgpackPDBDataset(
                self.split_df[self.split_df[mode] == True],
                self.lmdb_path,
                max_n_lig_patches=self.config.mol_encoder.n_patches,
                ligands=self.config.task.ligands,
                subsampling=subsampling,
                template_key=template_key,
            )
        else:
            return None

    def train_dataloader(self) -> DataLoader:
        # Use shuffle=False to ensure dataloader syncing in DDP mode
        return self._get_dataloader("train", self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val", self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test", self.test_ds, shuffle=False)

    def _get_dataloader(self, mode, dataset, **kwargs):
        if mode == "train":
            batch_size = self.batch_size
            epoch_frac = self.epoch_frac
        else:
            batch_size = 1
            epoch_frac = 1
        # dataset = self.datasets[mode]
        # sample_weights = dataset.weights / np.sum(dataset.weights)
        sampled_indices = np.random.choice(
            len(dataset),
            int(len(dataset) * epoch_frac),
            # p=sample_weights,
            replace=False,
        )
        if mode == "val":
            sampled_indices = np.repeat(sampled_indices, self.config.gpus)
        # print([dataset.keys[idx] for idx in sampled_indices[:20]])
        subdataset = torch.utils.data.Subset(dataset, sampled_indices)
        # subdataset = dataset
        return DataLoader(
            subdataset,
            batch_size=None,
            collate_fn=lambda x: _dynamic_batching_by_max_edgecount(
                x, self.config.task.edge_crop_size, batch_size
            ),
            num_workers=4,
            **kwargs,
        )


def _dynamic_batching_by_max_edgecount(x, max_n_edges, max_batch_size):
    if "num_u" in x["metadata"].keys():
        num_edges_upperbound = (
            x["metadata"]["num_a"] * 128 + x["metadata"]["num_i"] * 8 + 160**2
        )
    else:
        num_edges_upperbound = x["metadata"]["num_a"] * 128 + 160**2
    batch_size = max(1, min(max_n_edges // num_edges_upperbound, max_batch_size))
    return collate_samples([x] * batch_size)


@deprecated
class MsgpackPDBDataset(Dataset, ABC):
    def __init__(self, split_df, crop_size=None, ligands=True, mode=None):
        super(MsgpackPDBDataset, self).__init__()
        self.mode = mode
        self.negative_sampling_rate = 0.4
        self.crop_size = crop_size
        self.ligands = ligands
        self.split_df = split_df
        keys = split_df["protein_processed_path"].unique()
        if ligands:
            keys = split_df["protein_processed_path"].unique()
            self.keys = []
            for key in keys:
                if not os.path.exists(key):
                    continue
                self.keys.append(key)
            print(self.keys[:10])
        else:
            if len(split_df) > 30000:
                keys = split_df["protein_processed_path"].sample(n=9600).unique()
            else:
                keys = split_df["protein_processed_path"].unique()
            self.keys = []
            for key in keys:
                if not os.path.exists(key):
                    continue
                self.keys.append(key)
            print(self.keys[:10])

    def __len__(self):
        return len(self.keys)

    def _get_cropped_sample(self, index):
        try:
            sample_name = self.keys[index].split("/")[-1].split(".")[0]
            with self.env.begin(write=False) as txn:
                prot_sample = msgpack.unpackb(txn.get(sample_name.encode()))
                if self.ligands:
                    # Add indexer to ligand graphs and merge sample
                    native_ligand_samples = [
                        msgpack.unpackb(
                            txn.get(v.split("/")[-1].split(".")[0].encode())
                        )
                        for v in self.split_df[
                            self.split_df["protein_processed_path"] == self.keys[index]
                        ]["ligand_processed_path"]
                        if v is not None
                    ]
                    sample = merge_protein_and_ligands(
                        native_ligand_samples,
                        prot_sample,
                        edge_crop_size=self.edge_crop_size,
                    )
                else:
                    sample = crop_protein_features(prot_sample, self.crop_size)
                sample["metadata"]["sample_ID"] = sample_name
            return sample
        except:
            print(f"Warning: cropping failed for {sample_name}, retrying")
            return self._get_cropped_sample(index)

    def __getitem__(self, index):
        prot_sample = load_msgpack(self.keys[index])
        if self.ligands:
            ligand_samples = [
                load_msgpack(v)
                for v in self.split_df[
                    self.split_df["protein_processed_path"] == self.keys[index]
                ]["ligand_processed_path"]
                if v is not None
            ]
            sample = merge_protein_and_ligands(ligand_samples, prot_sample)
        else:
            sample = crop_protein_features(prot_sample, self.crop_size)
        sample_name = self.keys[index].split("/")[-1].split(".")[0]
        sample["metadata"]["sample_ID"] = sample_name
        return inplace_to_torch(sample)


class LMDBMsgpackPDBDataset(Dataset, ABC):
    def __init__(
        self,
        split_df,
        db_path,
        max_n_lig_patches=None,
        ligands=True,
        edge_crop_size=None,
        subsampling=1,
        template_key=None,
        filtering=True,
    ):
        super(LMDBMsgpackPDBDataset, self).__init__()
        self.max_n_lig_patches = max_n_lig_patches
        self.edge_crop_size = edge_crop_size
        self.env = lmdb.open(
            db_path, max_readers=32, readonly=True, lock=False, readahead=False
        )
        self.split_df = split_df.set_index("id")
        self.ligands = ligands
        self.template_key = template_key
        ds_length = len(split_df["id"].unique())
        keys = (
            split_df["id"]
            .drop_duplicates()
            .sample(n=(ds_length // subsampling), random_state=42)
            .tolist()
        )
        print(keys[:10])
        self.keys = []
        self.weights = []
        print("Tabulating sample IDs")
        for key in tqdm(keys):
            if filtering:
                try:
                    self.load_from_lmdb(key, featurize=False)
                except Exception as e:
                    print(f"Loading failed for {key}, dropping:", e)
                    continue
            # sampling_weight = self.split_df.loc[key]["training_sampling_weight"]
            self.keys.append(key)
            # self.weights.append(sampling_weight)
        print(self.keys[:10])

    def __len__(self):
        return len(self.keys)

    def load_from_lmdb(self, sample_name, featurize=True):
        with self.env.begin(write=False) as txn:
            df_row = self.split_df.loc[sample_name]
            prot_sample_id = df_row["protein_sample_id"]
            prot_sample = msgpack.unpackb(txn.get(prot_sample_id.encode()))
            if prot_sample["metadata"]["num_a"] > 1200:
                raise ValueError(
                    f'Too long sequence: {prot_sample["metadata"]["num_a"]}'
                )
            if self.ligands:
                if not df_row["ligands_sample_id"]:
                    native_ligand_samples = []
                else:
                    native_ligand_samples = [
                        msgpack.unpackb(txn.get(v.encode()))
                        for v in df_row["ligands_sample_id"].split(",")
                    ]
                if not featurize:
                    return None
                # Add indexer to ligand graphs and merge sample
                sample = merge_protein_and_ligands(
                    native_ligand_samples,
                    prot_sample,
                    self.max_n_lig_patches,
                    filter_ligands=False,
                    subsample_frames=False,
                )
            else:
                sample = crop_protein_features(prot_sample, self.crop_size)
            if sample["metadata"]["num_a"] + sample["metadata"]["num_i"] > 1250:
                raise ValueError(f'Too long sequence: {sample["metadata"]["num_a"]}')
            if self.template_key is not None:
                if self.template_key == "self":
                    template_sample = msgpack.unpackb(txn.get(prot_sample_id.encode()))
                else:
                    template_sample_names = df_row[self.template_key].split(",")
                    template_sample_name = random.choice(template_sample_names)
                    template_sample = msgpack.unpackb(
                        txn.get(template_sample_name.encode())
                    )
                sample = process_template_protein_features(sample, template_sample)
            sample["metadata"]["sample_ID"] = sample_name
        return sample

    def __getitem__(self, index):
        sample_name = self.keys[index]
        return inplace_to_torch(self.load_from_lmdb(sample_name))


class PDBDataset(Dataset, ABC):
    def __init__(
        self,
        dataset_path,
        metadata_path,
        split_ids,
        label_name,
        tokenizer=None,
        pretraining=False,
    ):
        super(PDBDataset, self).__init__()
        self.label_name = label_name
        self.metadata_path = metadata_path
        self.dataset_path = dataset_path
        self.df = pd.read_csv(
            self.metadata_path, keep_default_na=False
        )  # .set_index("pdb_id")
        self.ids = split_ids
        self.tokenizer = tokenizer
        self.token_only = False if self.tokenizer is None else True
        self._process(pretraining=pretraining)

    def _process(self, pretraining=False):
        if pretraining:
            self._process_pretraining()
        else:
            self._process_finetuning()

    def _process_pretraining(self):
        self.samples = []
        for index in tqdm(self.ids):
            row = self.df[self.df["index"] == index].iloc[0]
            code = row["pdb_id"]
            try:
                processed_path = os.path.join(
                    self.dataset_path, f"preprocessed_052522/{code}_merged.msgpack"
                )
                merged_sample = load_msgpack(processed_path)
                merged_sample["metadata"]["sample_ID"] = code
                self.samples.append(merged_sample)
            except Exception as e:
                print("Failed:", code, e)

    def _process_finetuning(self):
        self.samples = []
        for index in tqdm(self.ids):
            row = self.df[self.df["index"] == index].iloc[0]
            code = row["pdb_id"]
            try:
                try:
                    lig_sample = process_mol_file(
                        os.path.join(self.dataset_path, row["ligand_file"]),
                        tokenizer=self.tokenizer,
                    )
                except:
                    lig_sample = process_mol_file(
                        os.path.join(
                            self.dataset_path, row["ligand_file"].rstrip("sdf") + "mol2"
                        ),
                        tokenizer=self.tokenizer,
                    )
                ref_rec_sample = process_pdb(
                    pdb_string=open(
                        os.path.join(self.dataset_path, row["protein_file"])
                    ).read(),
                    chain_id=row["chain_id"],
                    res_start=row["begin_res_pdb"],
                    res_end=row["end_res_pdb"],
                    no_indexer=True,
                )
                rec_sample = process_pdb(
                    pdb_string=gzip.open(
                        os.path.join(self.dataset_path, row["afdb_file"])
                    )
                    .read()
                    .decode(),
                    chain_id=0,
                    res_start=row["begin_res_afdb"],
                    res_end=row["end_res_afdb"],
                )
                merged_sample = merge_protein_and_ligands(
                    lig_sample,
                    rec_sample,
                    label=None,
                    token_only=self.token_only,
                )
                merged_sample["metadata"]["sample_ID"] = index
                for key in merged_sample.keys():
                    for subkey, value in ref_rec_sample[key].items():
                        merged_sample[key]["target_" + subkey] = value
                assert np.array_equal(
                    merged_sample["features"]["res_type"],
                    merged_sample["features"]["target_res_type"],
                )
                self.samples.append(merged_sample)
            except Exception as e:
                print("Failed:", code, e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return inplace_to_torch(self.samples[index])


class LBADataset(Dataset, ABC):
    def __init__(self, dataset_path, split_ids, label_name, tokenizer=None):
        super(LBADataset, self).__init__()
        self.dataset_path = dataset_path
        self.label_name = label_name
        self.ligand_path = os.path.join(dataset_path, "ligand_files")
        self.receptor_path = os.path.join(dataset_path, "receptor_files")
        self.metadata_path = os.path.join(dataset_path, "metadata.csv")
        self.df = pd.read_csv(self.metadata_path).set_index("pdb_code")
        self.ids = split_ids
        self.tokenizer = tokenizer
        self.token_only = False if self.tokenizer is None else True
        self._process()

    def _process(self):
        self.samples = []
        for code in tqdm(self.ids):
            try:
                rec_sample = process_pdb(
                    open(
                        os.path.join(
                            self.dataset_path, self.df.loc[code]["receptor_file"]
                        )
                    ).read()
                )
                lig_sample = process_sdf(
                    Chem.SDMolSupplier(
                        os.path.join(
                            self.dataset_path, self.df.loc[code]["ligand_file"]
                        )
                    ),
                    tokenizer=self.tokenizer,
                )
                merged_sample = merge_protein_and_ligands(
                    lig_sample,
                    rec_sample,
                    label=self.df.loc[code][self.label_name],
                    token_only=self.token_only,
                )
                merged_sample["metadata"]["sample_ID"] = code
                self.samples.append(merged_sample)
            except Exception as e:
                print("Failed:", code, e)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return inplace_to_torch(self.samples[index])
