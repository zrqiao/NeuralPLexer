import torch

from neuralplexer.model.common import segment_mean


class LatentCoordinateConverter:
    """Transform the batched feature dict to latent coordinate arrays."""

    def __init__(self, config, prot_atom37_namemap, lig_namemap):
        super(LatentCoordinateConverter, self).__init__()
        self.config = config
        self.prot_namemap = prot_atom37_namemap
        self.lig_namemap = lig_namemap
        self.cached_noise = None
        self._last_pred_ca_trace = None

    @staticmethod
    def nested_get(dic, keys):
        for key in keys:
            dic = dic[key]
        return dic

    @staticmethod
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    def to_latent(self, batch):
        return None

    def assign_to_batch(self, batch, x_int):
        return None


class DefaultPLCoordinateConverter(LatentCoordinateConverter):
    """Minimal conversion, using internal coords for sidechains and global coords for others"""

    def __init__(self, config, prot_atom37_namemap, lig_namemap):
        super(DefaultPLCoordinateConverter, self).__init__(
            config, prot_atom37_namemap, lig_namemap
        )
        # Scale parameters in Angstrom
        self.ca_scale = config.global_max_sigma
        self.other_scale = config.internal_max_sigma
        self.ca_decay_rate = config.sde_decay_rate

    def to_latent(self, batch: dict):
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        self._batch_size = metadata["num_structid"]
        atom37_mask = batch["features"]["res_atom_mask"].bool()
        self._cother_mask = atom37_mask.clone()
        self._cother_mask[:, 1] = False
        atom37_coords = self.nested_get(batch, self.prot_namemap[0])
        ca_coords_glob = atom37_coords[:, 1].contiguous().view(self._batch_size, -1, 3)
        cother_coords_int = (
            (atom37_coords - atom37_coords[:, 1:2])[self._cother_mask]
            .contiguous()
            .view(self._batch_size, -1, 3)
        )
        self._n_res_per_sample = ca_coords_glob.shape[1]
        self._n_cother_per_sample = cother_coords_int.shape[1]
        if batch["misc"]["protein_only"]:
            self._n_ligha_per_sample = 0
            x_int = torch.cat(
                [ca_coords_glob / self.ca_scale, cother_coords_int / self.other_scale],
                dim=1,
            )
            lambdas = torch.cat(
                [
                    torch.full_like(ca_coords_glob, self.ca_decay_rate),
                    torch.full_like(
                        cother_coords_int,
                        self.ca_decay_rate * (self.ca_scale / self.other_scale),
                    ),
                ],
                dim=1,
            )
            return x_int, lambdas
        lig_ha_coords = self.nested_get(batch, self.lig_namemap[0])
        lig_centroid_coords = segment_mean(
            lig_ha_coords, indexer["gather_idx_i_molid"], metadata["num_molid"]
        )
        lig_ha_coords_int = (
            lig_ha_coords - lig_centroid_coords[indexer["gather_idx_i_molid"]]
        )
        lig_ha_coords_int = lig_ha_coords_int.contiguous().view(self._batch_size, -1, 3)
        lig_centroid_coords = lig_centroid_coords.contiguous().view(
            self._batch_size, -1, 3
        )
        x_int = torch.cat(
            [
                ca_coords_glob / self.ca_scale,
                cother_coords_int / self.other_scale,
                lig_centroid_coords / self.ca_scale,
                lig_ha_coords_int / self.other_scale,
            ],
            dim=1,
        )
        lambdas = torch.cat(
            [
                torch.full_like(ca_coords_glob, self.ca_decay_rate),
                torch.full_like(
                    cother_coords_int,
                    self.ca_decay_rate * (self.ca_scale / self.other_scale),
                ),
                torch.full_like(lig_centroid_coords, self.ca_decay_rate),
                torch.full_like(
                    lig_ha_coords_int,
                    self.ca_decay_rate * (self.ca_scale / self.other_scale),
                ),
            ],
            dim=1,
        )
        self._n_molid_per_sample = lig_centroid_coords.shape[1]
        self._n_ligha_per_sample = lig_ha_coords_int.shape[1]
        return x_int, lambdas

    def assign_to_batch(self, batch: dict, x_lat: torch.Tensor):
        indexer = batch["indexer"]
        new_atom37_coords = x_lat.new_zeros(
            self._batch_size * self._n_res_per_sample, 37, 3
        )
        if batch["misc"]["protein_only"]:
            ca_lat, cother_lat = torch.split(
                x_lat,
                [self._n_res_per_sample, self._n_cother_per_sample],
                dim=1,
            )
        else:
            ca_lat, cother_lat, ligcent_lat, lig_lat = torch.split(
                x_lat,
                [
                    self._n_res_per_sample,
                    self._n_cother_per_sample,
                    self._n_molid_per_sample,
                    self._n_ligha_per_sample,
                ],
                dim=1,
            )
        new_ca_glob = (ca_lat * self.ca_scale).contiguous().flatten(0, 1)
        new_atom37_coords[self._cother_mask] = (
            (cother_lat * self.other_scale).contiguous().flatten(0, 1)
        )
        new_atom37_coords = new_atom37_coords + new_ca_glob[:, None]
        new_atom37_coords[~self._cother_mask] = 0
        new_atom37_coords[:, 1] = new_ca_glob
        self.nested_set(batch, self.prot_namemap[1], new_atom37_coords)
        if batch["misc"]["protein_only"]:
            self.nested_set(batch, self.lig_namemap[1], None)
            self.empty_cache()
            return batch
        new_ligha_coords_int = (lig_lat * self.other_scale).contiguous().flatten(0, 1)
        new_ligha_coords_cent = (ligcent_lat * self.ca_scale).contiguous().flatten(0, 1)
        new_ligha_coords = (
            new_ligha_coords_int + new_ligha_coords_cent[indexer["gather_idx_i_molid"]]
        )
        self.nested_set(batch, self.lig_namemap[1], new_ligha_coords)
        self.empty_cache()
        return batch

    def empty_cache(self):
        self._batch_size = None
        self._cother_mask = None
        self._n_res_per_sample = None
        self._n_cother_per_sample = None
        self._n_ligha_per_sample = None
        self._n_molid_per_sample = None


class NullPLCoordinateConverter(LatentCoordinateConverter):
    """No conversion"""

    def __init__(self, config, prot_atom37_namemap, lig_namemap):
        super(NullPLCoordinateConverter, self).__init__(
            config, prot_atom37_namemap, lig_namemap
        )
        # Scale parameters in Angstrom
        self.ca_scale = 10.0
        self.other_scale = 10.0
        self.ca_decay_rate = config.sde_decay_rate

    def to_latent(self, batch: dict):
        batch["indexer"]
        metadata = batch["metadata"]
        self._batch_size = metadata["num_structid"]
        atom37_mask = batch["features"]["res_atom_mask"].bool()
        self._cother_mask = atom37_mask.clone()
        self._cother_mask[:, 1] = False
        atom37_coords = self.nested_get(batch, self.prot_namemap[0])
        ca_coords_glob = atom37_coords[:, 1].contiguous().view(self._batch_size, -1, 3)
        cother_coords_glob = (
            atom37_coords[self._cother_mask].contiguous().view(self._batch_size, -1, 3)
        )
        self._n_res_per_sample = ca_coords_glob.shape[1]
        self._n_cother_per_sample = cother_coords_glob.shape[1]
        if batch["misc"]["protein_only"]:
            self._n_ligha_per_sample = 0
            x_int = torch.cat(
                [ca_coords_glob / self.ca_scale, cother_coords_glob / self.other_scale],
                dim=1,
            )
            lambdas = torch.cat(
                [
                    torch.full_like(ca_coords_glob, self.ca_decay_rate),
                    torch.full_like(
                        cother_coords_glob,
                        self.ca_decay_rate * (self.ca_scale / self.other_scale),
                    ),
                ],
                dim=1,
            )
            return x_int, lambdas
        lig_ha_coords = self.nested_get(batch, self.lig_namemap[0])
        lig_ha_coords = lig_ha_coords.contiguous().view(self._batch_size, -1, 3)
        x_int = torch.cat(
            [
                ca_coords_glob / self.ca_scale,
                cother_coords_glob / self.other_scale,
                lig_ha_coords / self.other_scale,
            ],
            dim=1,
        )
        lambdas = torch.cat(
            [
                torch.full_like(ca_coords_glob, self.ca_decay_rate),
                torch.full_like(
                    cother_coords_glob,
                    self.ca_decay_rate * (self.ca_scale / self.other_scale),
                ),
                torch.full_like(lig_ha_coords, self.ca_decay_rate),
            ],
            dim=1,
        )
        self._n_ligha_per_sample = lig_ha_coords.shape[1]
        return x_int, lambdas

    def assign_to_batch(self, batch: dict, x_lat: torch.Tensor):
        new_atom37_coords = x_lat.new_zeros(
            self._batch_size * self._n_res_per_sample, 37, 3
        )
        if batch["misc"]["protein_only"]:
            ca_lat, cother_lat = torch.split(
                x_lat,
                [self._n_res_per_sample, self._n_cother_per_sample],
                dim=1,
            )
        else:
            ca_lat, cother_lat, lig_lat = torch.split(
                x_lat,
                [
                    self._n_res_per_sample,
                    self._n_cother_per_sample,
                    self._n_ligha_per_sample,
                ],
                dim=1,
            )
        new_ca_glob = (ca_lat * self.ca_scale).contiguous().flatten(0, 1)
        new_atom37_coords[self._cother_mask] = (
            (cother_lat * self.other_scale).contiguous().flatten(0, 1)
        )
        new_atom37_coords[~self._cother_mask] = 0
        new_atom37_coords[:, 1] = new_ca_glob
        self.nested_set(batch, self.prot_namemap[1], new_atom37_coords)
        if batch["misc"]["protein_only"]:
            self.nested_set(batch, self.lig_namemap[1], None)
            self.empty_cache()
            return batch
        new_ligha_coords = (lig_lat * self.other_scale).contiguous().flatten(0, 1)
        self.nested_set(batch, self.lig_namemap[1], new_ligha_coords)
        self.empty_cache()
        return batch

    def empty_cache(self):
        self._batch_size = None
        self._cother_mask = None
        self._n_res_per_sample = None
        self._n_cother_per_sample = None
        self._n_ligha_per_sample = None


class BranchedGaussianChainConverter(LatentCoordinateConverter):
    """Ideal chain model treating ligand as branched residues"""

    def __init__(self, config, prot_atom37_namemap, lig_namemap):
        super(BranchedGaussianChainConverter, self).__init__(
            config, prot_atom37_namemap, lig_namemap
        )
        # Scale parameters in Angstrom (diffusion coefficients)
        self.ca_scale = 1.0
        self.other_scale = 1.0
        self.ca_decay_rate = config.sde_decay_rate
        self.other_decay_rate = config.sde_decay_rate
        self.lig_res_anchor_mask = None

    def to_latent(self, batch: dict):
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        self._batch_size = metadata["num_structid"]
        atom37_mask = batch["features"]["res_atom_mask"].bool()
        self._cother_mask = atom37_mask.clone()
        self._cother_mask[:, 1] = False
        atom37_coords = self.nested_get(batch, self.prot_namemap[0])
        ca_coords_glob = atom37_coords[:, 1].contiguous().view(self._batch_size, -1, 3)
        cother_coords_int = (
            (atom37_coords - atom37_coords[:, 1:2])[self._cother_mask]
            .contiguous()
            .view(self._batch_size, -1, 3)
        )
        ca_coords_int = ca_coords_glob.clone()
        ca_coords_int[:, 1:] = ca_coords_int[:, 1:] - ca_coords_int[:, :-1]
        self._n_res_per_sample = ca_coords_glob.shape[1]
        self._n_cother_per_sample = cother_coords_int.shape[1]
        if batch["misc"]["protein_only"]:
            self._n_ligha_per_sample = 0
            x_int = torch.cat(
                [ca_coords_int / self.ca_scale, cother_coords_int / self.other_scale],
                dim=1,
            )
            lambdas = torch.cat(
                [
                    torch.full_like(ca_coords_int, self.ca_decay_rate),
                    torch.full_like(
                        cother_coords_int,
                        self.other_decay_rate,
                    ),
                ],
                dim=1,
            )
            return x_int, lambdas
        assert self.lig_res_anchor_mask is not None
        lig_ha_coords = self.nested_get(batch, self.lig_namemap[0])
        lig_ha_coords = lig_ha_coords.contiguous().view(self._batch_size, -1, 3)
        assert torch.sum(self.lig_res_anchor_mask) == metadata["num_molid"]
        anchor_res_coords = (
            ca_coords_glob[:, None, :, :]
            .expand(-1, max(metadata["num_molid_per_sample"]), -1, -1)
            .contiguous()[self.lig_res_anchor_mask]
        )
        lig_coords_int = lig_ha_coords - anchor_res_coords.contiguous()[
            indexer["gather_idx_i_molid"]
        ].contiguous().view(self._batch_size, -1, 3)
        x_int = torch.cat(
            [
                ca_coords_int / self.ca_scale,
                cother_coords_int / self.other_scale,
                lig_coords_int / self.other_scale,
            ],
            dim=1,
        )
        lambdas = torch.cat(
            [
                torch.full_like(ca_coords_int, self.ca_decay_rate),
                torch.full_like(
                    cother_coords_int,
                    self.other_decay_rate,
                ),
                torch.full_like(lig_coords_int, self.other_decay_rate),
            ],
            dim=1,
        )
        self._n_ligha_per_sample = lig_ha_coords.shape[1]
        return x_int, lambdas

    def assign_to_batch(self, batch: dict, x_lat: torch.Tensor):
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        new_atom37_coords = x_lat.new_zeros(
            self._batch_size * self._n_res_per_sample, 37, 3
        )
        if batch["misc"]["protein_only"]:
            ca_lat, cother_lat = torch.split(
                x_lat,
                [self._n_res_per_sample, self._n_cother_per_sample],
                dim=1,
            )
        else:
            ca_lat, cother_lat, lig_lat = torch.split(
                x_lat,
                [
                    self._n_res_per_sample,
                    self._n_cother_per_sample,
                    self._n_ligha_per_sample,
                ],
                dim=1,
            )
        new_ca_int = ca_lat * self.ca_scale
        new_ca_glob = torch.cumsum(new_ca_int, dim=1).contiguous().flatten(0, 1)
        new_atom37_coords[self._cother_mask] = (
            (cother_lat * self.other_scale).contiguous().flatten(0, 1)
        )
        new_atom37_coords = new_atom37_coords + new_ca_glob[:, None]
        new_atom37_coords[~self._cother_mask] = 0
        new_atom37_coords[:, 1] = new_ca_glob
        self.nested_set(batch, self.prot_namemap[1], new_atom37_coords)
        if batch["misc"]["protein_only"]:
            self.nested_set(batch, self.lig_namemap[1], None)
            self.empty_cache()
            return batch
        new_ligha_coords_int = (lig_lat * self.other_scale).contiguous().flatten(0, 1)
        new_anchor_res_coords = (
            new_ca_glob.contiguous()
            .view(self._batch_size, -1, 3)[:, None, :, :]
            .expand(-1, max(metadata["num_molid_per_sample"]), -1, -1)
            .contiguous()[self.lig_res_anchor_mask]
        )
        new_lig_coords_glob = (
            new_ligha_coords_int
            + new_anchor_res_coords.contiguous()[indexer["gather_idx_i_molid"]]
        )
        self.nested_set(batch, self.lig_namemap[1], new_lig_coords_glob)
        self.empty_cache()
        return batch

    def empty_cache(self):
        self._batch_size = None
        self._cother_mask = None
        self._n_res_per_sample = None
        self._n_cother_per_sample = None
        self._n_ligha_per_sample = None
