import random

import esm
import pytorch_lightning as pl
import scipy.stats
import tqdm
import wandb
from pytorch3d.ops import corresponding_points_alignment

from neuralplexer.model.config import get_base_config, get_standard_aa_features
from neuralplexer.model.cpm import BindingFormer, ProtFormer
from neuralplexer.model.esdm import EquivariantStructureDenoisingModule
from neuralplexer.model.mht_encoder import (MolPretrainingWrapper,
                                            _resolve_ligand_encoder)
from neuralplexer.util.sde_transform import (BranchedGaussianChainConverter,
                                             DefaultPLCoordinateConverter,
                                             NullPLCoordinateConverter)

BASE_CONFIG = get_base_config()
from neuralplexer.data.physical import (get_vdw_radii_array,
                                        get_vdw_radii_array_uff)
from neuralplexer.data.pipeline import (collate_numpy, inplace_to_device,
                                        inplace_to_torch)
from neuralplexer.model.modules import *
from neuralplexer.util.frame import (apply_similarity_transform,
                                     cartesian_to_internal, get_frame_matrix,
                                     internal_to_cartesian)


class NeuralPlexer(pl.LightningModule):
    def __init__(self, config=BASE_CONFIG):
        super(NeuralPlexer, self).__init__()
        self.config = config
        self.ligand_config = config.mol_encoder
        self.protein_config = config.protein_encoder
        self.contact_config = config.contact_predictor
        self.global_config = config.task
        self.protatm_padding_dim = self.protein_config.atom_padding_dim  # =37
        self.max_n_edges = self.global_config.edge_crop_size

        # VDW radius mapping, in Angstrom
        self.atnum2vdw = nn.Parameter(
            torch.tensor(get_vdw_radii_array() / 100.0),
            requires_grad=False,
        )
        self.atnum2vdw_uff = nn.Parameter(
            torch.tensor(get_vdw_radii_array_uff() / 100.0),
            requires_grad=False,
        )

        # Graph hyperparameters
        self.BINDING_SITE_CUTOFF = 6.0
        self.INTERNAL_TARGET_VARIANCE_SCALE = self.global_config.internal_max_sigma
        self.GLOBAL_TARGET_VARIANCE_SCALE = self.global_config.global_max_sigma
        self.CONTACT_SCALE = 5.0  # Fixed hyperparameter
        self.max_rescaled_sigma = 5.0  # Estimated data variance

        (
            standard_aa_template_featset,
            standard_aa_graph_featset,
        ) = get_standard_aa_features()
        self._standard_aa_template_featset = inplace_to_torch(
            standard_aa_template_featset
        )
        self._standard_aa_molgraph_featset = inplace_to_torch(
            collate_numpy(standard_aa_graph_featset)
        )

        node_dim = self.protein_config.residue_dim
        self.ligand_encoder = _resolve_ligand_encoder(
            self.ligand_config, self.global_config
        )
        self.lig_masking_rate = self.global_config.max_masking_rate
        # Protein sequence language model
        if self.protein_config.use_esm_embedding:
            self.plm, plm_alphabet = esm.pretrained.load_model_and_alphabet_hub(
                self.protein_config.esm_version
            )
            self.esm_repr_layer = self.protein_config.esm_repr_layer
            self.batch_converter = plm_alphabet.get_batch_converter()
            self.plm_adapter = nn.Linear(self.plm.embed_dim, node_dim, bias=False)
            for p in self.plm.parameters():
                p.requires_grad = False
        else:
            self.res_in_projector = nn.Linear(
                self.protein_config.n_aa_types,
                node_dim,
                bias=False,
            )

        # Embed the diffused protein structure and templates
        self.protein_encoder = ProtFormer(
            node_dim,
            self.protein_config.pair_dim,
            n_heads=self.protein_config.n_heads,
            n_blocks=self.protein_config.n_encoder_stacks,
            n_protein_patches=self.protein_config.n_patches,
            dropout=self.global_config.dropout,
        )

        # Relational reasoning and contact prediction module
        self.molgraph_single_projector = nn.Linear(
            self.ligand_config.node_channels, node_dim, bias=False
        )
        self.molgraph_pair_projector = nn.Linear(
            self.ligand_config.pair_channels, self.protein_config.pair_dim, bias=False
        )
        self.covalent_embed = nn.Embedding(2, self.protein_config.pair_dim)
        self.contact_code_embed = nn.Embedding(2, self.protein_config.pair_dim)

        self.pl_contact_stack = BindingFormer(
            node_dim,
            self.protein_config.pair_dim,
            n_heads=self.protein_config.n_heads,
            n_blocks=self.contact_config.n_stacks,
            n_protein_patches=self.protein_config.n_patches,
            n_ligand_patches=self.ligand_config.n_patches,
            dropout=self.global_config.dropout,
        )
        # Distogram heads
        self.dist_bins = nn.Parameter(torch.linspace(2, 22, 32), requires_grad=False)
        self.dgram_head = GELUMLP(
            self.protein_config.pair_dim,
            32,
            n_hidden_feats=self.protein_config.pair_dim,
            zero_init=True,
        )
        self.score_head = EquivariantStructureDenoisingModule(
            config.score_head.fiber_dim,
            input_dim=node_dim,
            input_pair_dim=self.protein_config.pair_dim,
            hidden_dim=config.score_head.hidden_dim,
            n_stacks=config.score_head.n_stacks,
            n_heads=self.protein_config.n_heads,
            dropout=self.global_config.dropout,
        )
        self.confidence_head = EquivariantStructureDenoisingModule(
            config.score_head.fiber_dim,
            input_dim=node_dim,
            input_pair_dim=self.protein_config.pair_dim,
            hidden_dim=config.score_head.hidden_dim,
            n_stacks=config.score_head.n_stacks,
            n_heads=self.protein_config.n_heads,
            dropout=self.global_config.dropout,
        )
        self.plddt_gram_head = GELUMLP(
            self.protein_config.pair_dim,
            8,
            n_hidden_feats=self.protein_config.pair_dim,
            zero_init=True,
        )
        self.contact_encoding = nn.Embedding(2, node_dim)

        self._freeze_pretraining_params()
        self.save_hyperparameters()

    def _freeze_pretraining_params(self):
        if self.global_config.freeze_ligand_encoder:
            for p in self.ligand_encoder.parameters():
                p.requires_grad = False
        if self.global_config.freeze_protein_encoder:
            for module in [
                self.plm_adapter,
                self.protein_encoder,
            ]:
                for p in module.parameters():
                    p.requires_grad = False

    def forward(
        self,
        batch,
        iter_id=0,
        observed_block_contacts=None,
        contact_prediction=True,
        infer_geometry_prior=False,
        score=False,
        affinity=False,
        use_template=False,
        **kwargs,
    ):
        metadata = batch["metadata"]
        features = batch["features"]
        indexer = batch["indexer"]
        if "outputs" not in batch.keys():
            batch["outputs"] = {}

        batch = self._run_encoder_stack(
            batch,
            use_template=use_template,
            use_plddt=self.global_config.use_plddt,
            **kwargs,
        )

        if contact_prediction:
            self._run_contact_map_stack(
                batch,
                iter_id,
                observed_block_contacts=observed_block_contacts,
                **kwargs,
            )

        if infer_geometry_prior:
            assert batch["misc"]["protein_only"] == False
            self._infer_geometry_prior(batch, **kwargs)

        if score:
            batch["outputs"]["denoised_prediction"] = self._run_score_head(
                batch, embedding_iter_id=iter_id, **kwargs
            )

        if affinity:
            lig_diff = self.affinity_head(features["lig_atom_attr"])
            return self.pooling(
                lig_diff,
                indexer["gather_idx_i_molid"],
                metadata["num_molid"],
            ).squeeze(1)

        return batch

    def _extract_esm_embeddings(self, sequence_data, sequence_mask):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequence_data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.plm(batch_tokens, repr_layers=[self.esm_repr_layer])
        token_representations = results["representations"][self.esm_repr_layer]
        sequence_representations = []
        for i, (_, seq) in enumerate(sequence_data):
            sequence_representations.append(token_representations[i, 1 : len(seq) + 1])
        sequence_representations = torch.cat(sequence_representations, dim=0)
        res_plm_embeddings = sequence_representations[sequence_mask]
        return res_plm_embeddings

    def _initialize_protein_embeddings(self, batch):
        metadata = batch["metadata"]
        features = batch["features"]
        batch_size = metadata["num_structid"]

        # Protein residue and residue-pair embeddings
        if "res_embedding_in" not in features:
            if self.protein_config.use_esm_embedding:
                assert self.global_config.single_protein_batch
                # First sequence for monomers
                prot_stoi = max(batch["metadata"]["num_chainid_per_sample"])
                first_sample_seqlength = sum(
                    [len(v[1]) for v in batch["misc"]["sequence_data"][:prot_stoi]]
                )
                res_plm_embeddings = self._extract_esm_embeddings(
                    batch["misc"]["sequence_data"][:prot_stoi],
                    features["sequence_res_mask"][:first_sample_seqlength].bool(),
                )
                features["res_embedding_in"] = (
                    self.plm_adapter(res_plm_embeddings)[None, :]
                    .expand(batch_size, -1, -1)
                    .flatten(0, 1)
                )
                assert (
                    features["res_embedding_in"].shape[0]
                    == features["res_atom_types"].shape[0]
                )
            else:
                features["res_embedding_in"] = self.res_in_projector(
                    F.one_hot(
                        features["res_type"].long(),
                        num_classes=self.protein_config.n_aa_types,
                    ).float()
                )

    def _initialize_ligand_embeddings(self, batch):
        metadata = batch["metadata"]
        batch["features"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]

        # Ligand atom, frame and pair embeddings
        if self.training:
            masking_rate = random.uniform(0, self.lig_masking_rate)
        else:
            masking_rate = 0
        batch = self.ligand_encoder(batch, masking_rate=masking_rate)
        batch["features"]["lig_atom_attr_projected"] = self.molgraph_single_projector(
            batch["features"]["lig_atom_attr"]
        )
        # Downsampled ligand frames
        batch["features"]["lig_trp_attr_projected"] = self.molgraph_single_projector(
            batch["features"]["lig_trp_attr"]
        )
        batch["features"][
            "lig_atom_pair_attr_projected"
        ] = self.molgraph_pair_projector(batch["features"]["lig_atom_pair_attr"])
        lig_af_pair_attr_flat_ = self.molgraph_pair_projector(
            batch["features"]["lig_af_pair_attr"]
        )
        batch["features"]["lig_af_pair_attr_projected"] = lig_af_pair_attr_flat_

        if self.global_config.single_protein_batch:
            lig_af_pair_attr = lig_af_pair_attr_flat_.new_zeros(
                batch_size,
                max(metadata["num_U_per_sample"]),
                max(metadata["num_I_per_sample"]),
                self.protein_config.pair_dim,
            )
            n_U_first = max(metadata["num_U_per_sample"])
            n_I_first = max(metadata["num_I_per_sample"])
            max(metadata["num_i_per_sample"])
            lig_af_pair_attr[
                indexer["gather_idx_UI_U"] // n_U_first,
                indexer["gather_idx_UI_U"] % n_U_first,
                indexer["gather_idx_UI_I"] % n_I_first,
            ] = lig_af_pair_attr_flat_

            batch["features"]["lig_af_grid_attr_projected"] = lig_af_pair_attr
        else:
            raise NotImplementedError

    def _init_randexp_kNN_edges_and_covmask(self, batch, detect_covalent=False):
        batch_size = batch["metadata"]["num_structid"]
        protatm_padding_mask = batch["features"]["res_atom_mask"]
        prot_atm_coords_padded = batch["features"]["input_protein_coords"]
        protatm_coords = prot_atm_coords_padded[protatm_padding_mask].contiguous()
        n_protatm_per_sample = batch["metadata"]["num_protatm_per_sample"]
        protatm_coords = protatm_coords.view(batch_size, n_protatm_per_sample, 3)
        if not batch["misc"]["protein_only"]:
            n_ligatm_per_sample = max(batch["metadata"]["num_i_per_sample"])
            ligatm_coords = batch["features"]["input_ligand_coords"]
            ligatm_coords = ligatm_coords.view(batch_size, n_ligatm_per_sample, 3)
            atm_coords = torch.cat([protatm_coords, ligatm_coords], dim=1)
        else:
            atm_coords = protatm_coords
        distance_mat = torch.norm(
            atm_coords[:, :, None] - atm_coords[:, None, :], dim=-1
        )
        distance_mat[distance_mat == 0] = 1e9
        knn_edge_mask = topk_edge_mask_from_logits(
            -distance_mat / self.CONTACT_SCALE,
            self.config.score_head.max_atom_degree,
            randomize=True,
        )
        if (not batch["misc"]["protein_only"]) and detect_covalent:
            prot_atomic_numbers = batch["features"]["protatm_to_atomic_number"].view(
                batch_size, n_protatm_per_sample
            )
            lig_atomic_numbers = (
                batch["features"]["atomic_numbers"]
                .long()
                .view(batch_size, n_ligatm_per_sample)
            )
            atom_vdw = self.atnum2vdw[
                torch.cat([prot_atomic_numbers, lig_atomic_numbers], dim=1)
            ]
            average_vdw = (atom_vdw[:, :, None] + atom_vdw[:, None, :]) / 2
            intermol_iscov_mask = distance_mat < average_vdw * 1.3
            intermol_iscov_mask[:, :n_protatm_per_sample, :n_protatm_per_sample] = False
            gather_idx_i_molid = batch["indexer"]["gather_idx_i_molid"].view(
                batch_size, n_ligatm_per_sample
            )
            lig_samemol_mask = (
                gather_idx_i_molid[:, :, None] == gather_idx_i_molid[:, None, :]
            )
            intermol_iscov_mask[
                :, n_protatm_per_sample:, n_protatm_per_sample:
            ] = intermol_iscov_mask[:, n_protatm_per_sample:, n_protatm_per_sample:] & (
                ~lig_samemol_mask
            )
            knn_edge_mask = knn_edge_mask | intermol_iscov_mask
        else:
            intermol_iscov_mask = torch.zeros_like(distance_mat, dtype=torch.bool)
        p_idx = torch.arange(
            batch_size * n_protatm_per_sample, device=self.device
        ).view(batch_size, n_protatm_per_sample)
        pp_edge_mask = knn_edge_mask[:, :n_protatm_per_sample, :n_protatm_per_sample]
        batch["indexer"]["knn_idx_protatm_protatm_src"] = (
            p_idx[:, :, None]
            .expand(-1, -1, n_protatm_per_sample)
            .contiguous()[pp_edge_mask]
        )
        batch["indexer"]["knn_idx_protatm_protatm_dst"] = (
            p_idx[:, None, :]
            .expand(-1, n_protatm_per_sample, -1)
            .contiguous()[pp_edge_mask]
        )
        batch["features"]["knn_feat_protatm_protatm"] = self.covalent_embed(
            intermol_iscov_mask[:, :n_protatm_per_sample, :n_protatm_per_sample][
                pp_edge_mask
            ].long()
        )
        if not batch["misc"]["protein_only"]:
            l_idx = torch.arange(
                batch_size * n_ligatm_per_sample, device=self.device
            ).view(batch_size, n_ligatm_per_sample)
            pl_edge_mask = knn_edge_mask[
                :, :n_protatm_per_sample, n_protatm_per_sample:
            ]
            batch["indexer"]["knn_idx_protatm_ligatm_src"] = (
                p_idx[:, :, None]
                .expand(-1, -1, n_ligatm_per_sample)
                .contiguous()[pl_edge_mask]
            )
            batch["indexer"]["knn_idx_protatm_ligatm_dst"] = (
                l_idx[:, None, :]
                .expand(-1, n_protatm_per_sample, -1)
                .contiguous()[pl_edge_mask]
            )
            batch["features"]["knn_feat_protatm_ligatm"] = self.covalent_embed(
                intermol_iscov_mask[:, :n_protatm_per_sample, n_protatm_per_sample:][
                    pl_edge_mask
                ].long()
            )
            lp_edge_mask = knn_edge_mask[
                :, n_protatm_per_sample:, :n_protatm_per_sample
            ]
            batch["indexer"]["knn_idx_ligatm_protatm_src"] = (
                l_idx[:, :, None]
                .expand(-1, -1, n_protatm_per_sample)
                .contiguous()[lp_edge_mask]
            )
            batch["indexer"]["knn_idx_ligatm_protatm_dst"] = (
                p_idx[:, None, :]
                .expand(-1, n_ligatm_per_sample, -1)
                .contiguous()[lp_edge_mask]
            )
            batch["features"]["knn_feat_ligatm_protatm"] = self.covalent_embed(
                intermol_iscov_mask[:, n_protatm_per_sample:, :n_protatm_per_sample][
                    lp_edge_mask
                ].long()
            )
            ll_edge_mask = knn_edge_mask[
                :, n_protatm_per_sample:, n_protatm_per_sample:
            ]
            batch["indexer"]["knn_idx_ligatm_ligatm_src"] = (
                l_idx[:, :, None]
                .expand(-1, -1, n_ligatm_per_sample)
                .contiguous()[ll_edge_mask]
            )
            batch["indexer"]["knn_idx_ligatm_ligatm_dst"] = (
                l_idx[:, None, :]
                .expand(-1, n_ligatm_per_sample, -1)
                .contiguous()[ll_edge_mask]
            )
            batch["features"]["knn_feat_ligatm_ligatm"] = self.covalent_embed(
                intermol_iscov_mask[:, n_protatm_per_sample:, n_protatm_per_sample:][
                    ll_edge_mask
                ].long()
            )

    def _run_encoder_stack(
        self,
        batch,
        **kwargs,
    ):
        with torch.no_grad():
            batch = self._prepare_protein_patch_indexers(
                batch, randomize_anchors=self.training
            )
            self._prepare_protein_backbone_indexers(batch)
            self._initialize_protein_embeddings(batch)
            self._initialize_protatm_indexer_and_embeddings(batch)

        batch = self.protein_encoder(
            batch,
            in_attr_suffix="",
            out_attr_suffix="_projected",
            **kwargs,
        )
        if batch["misc"]["protein_only"]:
            return batch

        # Assuming static ligand graph
        if "lig_atom_attr" not in batch["features"].keys():
            self._initialize_ligand_embeddings(batch)
        return batch

    def _run_contact_map_stack(
        self,
        batch,
        iter_id,
        observed_block_contacts=None,
        **kwargs,
    ):
        if observed_block_contacts is not None:
            # Merge into 8AA blocks and gather to patches
            patch8_idx = (
                torch.arange(
                    observed_block_contacts.shape[1],
                    device=self.device,
                )
                // 8
            )
            merged_contacts_reswise = (
                segment_sum(
                    observed_block_contacts.transpose(0, 1).contiguous(),
                    patch8_idx,
                    max(patch8_idx) + 1,
                )
                .bool()[patch8_idx]
                .transpose(0, 1)
                .contiguous()
            )
            merged_contacts_gathered = (
                merged_contacts_reswise.flatten(0, 1)[
                    batch["indexer"]["gather_idx_pid_a"]
                ]
                .contiguous()
                .view(
                    observed_block_contacts.shape[0],
                    -1,
                    observed_block_contacts.shape[2],
                )
            )
            block_contact_embedding = self.contact_code_embed(
                merged_contacts_gathered.long()
            )
        else:
            block_contact_embedding = None
        batch = self.pl_contact_stack(
            batch,
            in_attr_suffix="_projected",
            out_attr_suffix=f"_out_{iter_id}",
            observed_block_contacts=block_contact_embedding,
        )

        if batch["misc"]["protein_only"]:
            return batch

        metadata = batch["metadata"]
        batch_size = metadata["num_structid"]
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_I_per_sample = metadata["n_lig_patches_per_sample"]
        res_lig_pair_attr = batch["features"][f"res_trp_pair_attr_flat_out_{iter_id}"]
        raw_dgram_logits = self.dgram_head(res_lig_pair_attr).view(
            batch_size, n_a_per_sample, n_I_per_sample, 32
        )
        batch["outputs"][f"res_lig_distogram_out_{iter_id}"] = F.log_softmax(
            raw_dgram_logits, dim=-1
        )
        return batch

    def _merge_res_lig_logits_to_block(self, batch, res_lig_logits):
        # Patch merging [B, N_res, N_atm] -> [B, N_patch, N_graph]
        assert self.global_config.single_protein_batch
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]
        n_mol_per_sample = max(metadata["num_molid_per_sample"])
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_I_per_sample = max(metadata["num_I_per_sample"])
        res_lig_logits = res_lig_logits.view(batch_size, n_a_per_sample, n_I_per_sample)
        patch_wise_logits = segment_logsumexp(
            segment_logsumexp(
                res_lig_logits.transpose(0, 1),
                indexer["gather_idx_a_pid"][:n_a_per_sample],
                metadata["n_prot_patches_per_sample"],
            ).permute(2, 1, 0),
            indexer["gather_idx_I_molid"][:n_I_per_sample],
            n_mol_per_sample,
        ).permute(1, 2, 0)
        return patch_wise_logits

    def _merge_res_lig_logits_to_patch(self, batch, res_lig_logits):
        # Patch merging [B, N_res, N_atm] -> [B, N_patch, N_graph]
        assert self.global_config.single_protein_batch
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]
        max(metadata["num_molid_per_sample"])
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_A_per_sample = metadata["n_prot_patches_per_sample"]
        n_I_per_sample = max(metadata["num_I_per_sample"])
        res_lig_logits = res_lig_logits.view(batch_size, n_a_per_sample, n_I_per_sample)
        patch_wise_logits = (
            segment_logsumexp(
                res_lig_logits.contiguous().flatten(0, 1),
                indexer["gather_idx_a_pid"],
                n_A_per_sample * batch_size,
            )
            .contiguous()
            .view(batch_size, n_A_per_sample, n_I_per_sample)
        )
        return patch_wise_logits

    def _merge_res_lig_logits_to_graph(self, batch, res_lig_logits):
        # Patch merging [B, N_res, N_atm] -> [B, N_res, N_graph]
        assert self.global_config.single_protein_batch
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]
        n_mol_per_sample = max(metadata["num_molid_per_sample"])
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_I_per_sample = max(metadata["num_I_per_sample"])
        res_lig_logits = res_lig_logits.view(batch_size, n_a_per_sample, n_I_per_sample)
        graph_wise_logits = segment_logsumexp(
            res_lig_logits.permute(2, 0, 1),
            indexer["gather_idx_I_molid"][:n_I_per_sample],
            n_mol_per_sample,
        ).permute(1, 2, 0)
        return graph_wise_logits

    def _sample_block_contact_matrix(self, batch, res_lig_logits, last=None):
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]
        max(metadata["num_molid_per_sample"])
        n_A_per_sample = metadata["n_prot_patches_per_sample"]
        n_I_per_sample = max(metadata["num_I_per_sample"])
        # Sampling from unoccupied lattice sites
        patch_wise_logits = self._merge_res_lig_logits_to_patch(batch, res_lig_logits)
        if last is None:
            last = torch.zeros_like(patch_wise_logits, dtype=torch.bool)
        # Column-graph-wise masking for already sampled ligands
        # sampled_ligand_mask = torch.amax(last, dim=1, keepdim=True)
        sampled_ligand_mask = (
            segment_sum(
                torch.sum(last, dim=1).contiguous().flatten(0, 1),
                indexer["gather_idx_I_molid"],
                metadata["num_molid"],
            )[indexer["gather_idx_I_molid"]]
            .contiguous()
            .view(batch_size, 1, n_I_per_sample)
        )
        masked_logits = patch_wise_logits - sampled_ligand_mask * 1e9
        sampled_block_onehot = batched_sample_onehot(
            masked_logits.flatten(1, 2), dim=1
        ).view(batch_size, n_A_per_sample, n_I_per_sample)
        new_block_contact_mat = last + sampled_block_onehot
        return new_block_contact_mat

    def _sample_reslig_contact_matrix(self, batch, res_lig_logits, last=None):
        metadata = batch["metadata"]
        batch_size = metadata["num_structid"]
        max(metadata["num_molid_per_sample"])
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_I_per_sample = max(metadata["num_I_per_sample"])
        res_lig_logits = res_lig_logits.view(batch_size, n_a_per_sample, n_I_per_sample)
        # Sampling from unoccupied lattice sites
        if last is None:
            last = torch.zeros_like(res_lig_logits, dtype=torch.bool)
        # Column-graph-wise masking for already sampled ligands
        # sampled_ligand_mask = torch.amax(last, dim=1, keepdim=True)
        sampled_frame_mask = torch.sum(last, dim=1, keepdim=True).contiguous()
        masked_logits = res_lig_logits - sampled_frame_mask * 1e9
        sampled_block_onehot = batched_sample_onehot(
            masked_logits.flatten(1, 2), dim=1
        ).view(batch_size, n_a_per_sample, n_I_per_sample)
        new_block_contact_mat = last + sampled_block_onehot
        # Remove non-contact patches
        valid_logit_mask = res_lig_logits > -16.0
        new_block_contact_mat = (new_block_contact_mat * valid_logit_mask).bool()
        return new_block_contact_mat

    def _sample_res_rowmask_from_contacts(self, batch, res_ligatm_logits):
        metadata = batch["metadata"]
        max(metadata["num_molid_per_sample"])
        lig_wise_logits = (
            self._merge_res_lig_logits_to_graph(batch, res_ligatm_logits)
            .permute(0, 2, 1)
            .contiguous()
        )
        sampled_res_onehot_mask = batched_sample_onehot(
            lig_wise_logits.flatten(0, 1), dim=1
        )
        return sampled_res_onehot_mask

    def _infer_geometry_prior(
        self,
        batch,
        cached_block_contacts=None,
        binding_site_mask=None,
        logit_clamp_value=None,
        **kwargs,
    ):
        # Parse self.task_config.block_contact_decoding_scheme
        assert self.global_config.block_contact_decoding_scheme == "beam"
        n_lig_frames = max(batch["metadata"]["num_I_per_sample"])
        # Autoregressive block-contact sampling
        if cached_block_contacts is None:
            # Start from the prior distribution
            sampled_block_contacts = None
            last_distogram = batch["outputs"]["res_lig_distogram_out_0"]
            for iter_id in tqdm.tqdm(
                range(n_lig_frames), desc="Block contact sampling"
            ):
                last_contact_map = self._distogram_to_gaussian_contact_logits(
                    last_distogram
                )
                sampled_block_contacts = self._sample_reslig_contact_matrix(
                    batch, last_contact_map, last=sampled_block_contacts
                ).detach()
                self._run_contact_map_stack(
                    batch, iter_id, observed_block_contacts=sampled_block_contacts
                )
                last_distogram = batch["outputs"][f"res_lig_distogram_out_{iter_id}"]
            batch["outputs"]["sampled_block_contacts_last"] = sampled_block_contacts
            # Check that all ligands are assigned to one protein chain segment
            num_assigned_per_lig = segment_sum(
                torch.sum(sampled_block_contacts, dim=1).contiguous().flatten(0, 1),
                batch["indexer"]["gather_idx_I_molid"],
                batch["metadata"]["num_molid"],
            )
            assert torch.all(num_assigned_per_lig >= 1)
        else:
            sampled_block_contacts = cached_block_contacts

        # Use the cached contacts and only sample once
        self._run_contact_map_stack(
            batch, n_lig_frames, observed_block_contacts=sampled_block_contacts
        )
        last_distogram = batch["outputs"][f"res_lig_distogram_out_{n_lig_frames}"]
        res_lig_contact_logit_pred = self._distogram_to_gaussian_contact_logits(
            last_distogram
        )

        if binding_site_mask is not None:
            res_lig_contact_logit_pred = res_lig_contact_logit_pred - (
                ~binding_site_mask[:, :, None] * 1e9
            )
        if not self.training and logit_clamp_value is not None:
            res_lig_contact_logit_pred = (
                res_lig_contact_logit_pred
                - (res_lig_contact_logit_pred < logit_clamp_value) * 1e9
            )
        batch["outputs"]["geometry_prior_L"] = res_lig_contact_logit_pred.flatten()

    def _init_esdm_inputs(self, batch, embedding_iter_id):
        with torch.no_grad():
            self._init_randexp_kNN_edges_and_covmask(
                batch, detect_covalent=self.global_config.detect_covalent
            )
        batch["features"]["rec_res_attr_decin"] = batch["features"][
            f"rec_res_attr_out_{embedding_iter_id}"
        ]
        batch["features"]["res_res_pair_attr_decin"] = batch["features"][
            f"res_res_pair_attr_out_{embedding_iter_id}"
        ]
        batch["features"]["res_res_grid_attr_flat_decin"] = batch["features"][
            f"res_res_grid_attr_flat_out_{embedding_iter_id}"
        ]
        if batch["misc"]["protein_only"]:
            return batch
        batch["features"]["lig_trp_attr_decin"] = batch["features"][
            f"lig_trp_attr_out_{embedding_iter_id}"
        ]
        # Use protein-ligand edges from the contact predictor
        batch["features"]["res_trp_grid_attr_flat_decin"] = batch["features"][
            f"res_trp_grid_attr_flat_out_{embedding_iter_id}"
        ]
        batch["features"]["res_trp_pair_attr_flat_decin"] = batch["features"][
            f"res_trp_pair_attr_flat_out_{embedding_iter_id}"
        ]
        batch["features"]["trp_trp_grid_attr_flat_decin"] = batch["features"][
            f"trp_trp_grid_attr_flat_out_{embedding_iter_id}"
        ]
        return batch

    def _run_score_head(self, batch, embedding_iter_id, **kwargs):
        batch = self._init_esdm_inputs(batch, embedding_iter_id)
        return self.score_head(
            batch, frozen_prot=self.global_config.frozen_backbone, **kwargs
        )

    def _run_confidence_head(self, batch, **kwargs):
        batch = self._init_esdm_inputs(batch, "confidence")
        return self.confidence_head(batch, frozen_prot=False, **kwargs)

    def run_confidence_estimation(self, batch, struct, return_avg_stats=False):
        batch_size = batch["metadata"]["num_structid"]
        batch["features"]["input_protein_coords"] = struct["receptor_padded"].clone()
        if struct["ligands"] is not None:
            batch["features"]["input_ligand_coords"] = struct["ligands"].clone()
        else:
            batch["features"]["input_ligand_coords"] = None
        self._assign_timestep_encodings(batch, 0.0)
        batch = self._run_encoder_stack(batch, use_template=False, use_plddt=False)
        self._run_contact_map_stack(batch, iter_id="confidence")
        conf_out = self._run_confidence_head(batch)
        conf_rep = (
            conf_out["final_embedding_prot_res"][:, 0]
            .contiguous()
            .view(batch_size, -1, self.config.score_head.fiber_dim)
        )
        if struct["ligands"] is not None:
            conf_rep_lig = (
                conf_out["final_embedding_lig_atom"][:, 0]
                .contiguous()
                .view(batch_size, -1, self.config.score_head.fiber_dim)
            )
            conf_rep = torch.cat([conf_rep, conf_rep_lig], dim=1)
        plddt_logits = F.log_softmax(self.plddt_gram_head(conf_rep), dim=-1)
        batch["outputs"]["plddt_logits"] = plddt_logits
        plddt_gram = torch.exp(plddt_logits)
        batch["outputs"]["plddt"] = torch.cumsum(plddt_gram[:, :, :4], dim=-1).mean(
            dim=-1
        )

        if return_avg_stats:
            plddt_avg = (
                batch["outputs"]["plddt"].view(batch_size, -1).mean(dim=1)
            ).detach()
            if struct["ligands"] is not None:
                plddt_avg_lig = (
                    batch["outputs"]["plddt"]
                    .view(batch_size, -1)[:, batch["metadata"]["num_a_per_sample"][0] :]
                    .mean(dim=1)
                    .detach()
                )
            else:
                plddt_avg_lig = None
            return plddt_avg, plddt_avg_lig

        return batch

    def _get_epoch_outputs(self, outs):
        preds = [v["out"] for v in outs]
        labels = [v["labels"] for v in outs]
        preds = torch.cat(preds, dim=0).cpu().tolist()
        labels = torch.cat(labels, dim=0).cpu().tolist()
        return preds, labels

    def _compute_epoch_r2(self, outs):
        preds = [v["out"] for v in outs]
        labels = [v["labels"] for v in outs]
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        return self.r2score(preds, labels)

    def _compute_pearson_r(self, outs):
        preds = [v["out"] for v in outs]
        labels = [v["labels"] for v in outs]
        preds = torch.cat(preds, dim=0).cpu().numpy()
        labels = torch.cat(labels, dim=0).cpu().numpy()
        return scipy.stats.pearsonr(preds, labels)[0]

    def compute_template_weighted_centroid_drmsd(
        self,
        batch,
        pred_prot_coords,
    ):
        batch_size = batch["metadata"]["num_structid"]

        pred_cent_coords = (
            pred_prot_coords.mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
            .sum(dim=1)
            .div(batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9)
        ).view(batch_size, -1, 3)
        pred_dist = (
            torch.square(pred_cent_coords[:, :, None] - pred_cent_coords[:, None, :])
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
        with torch.no_grad():
            target_cent_coords = (
                batch["features"]["res_atom_positions"]
                .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
                .sum(dim=1)
                .div(
                    batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9
                )
            ).view(batch_size, -1, 3)
            template_cent_coords = (
                batch["features"]["template_atom_positions"]
                .mul(batch["features"]["template_atom37_mask"].bool()[:, :, None])
                .sum(dim=1)
                .div(
                    batch["features"]["template_atom37_mask"].bool().sum(dim=1)[:, None]
                    + 1e-9
                )
            ).view(batch_size, -1, 3)
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
                batch["features"]["template_alignment_mask"].bool().view(batch_size, -1)
            )
            motion_mask = (
                ((target_dist - template_dist).abs() > 2.0)
                * template_alignment_mask[:, None, :]
                * template_alignment_mask[:, :, None]
            )

        dist_errors = (pred_dist - target_dist).square()
        drmsd = (
            dist_errors.add(1e-4).sqrt().sub(1e-2).mul(motion_mask).sum(dim=(1, 2))
        ) / (motion_mask.long().sum(dim=(1, 2)) + 1)
        return drmsd

    def compute_drmsd_and_clashloss(
        self,
        batch,
        pred_prot_coords,
        target_prot_coords,
        cap_size=4000,
        pred_lig_coords=None,
        target_lig_coords=None,
        ligatm_types=None,
        binding_site=False,
        pl_interface=False,
    ):
        features = batch["features"]
        with torch.no_grad():
            if not binding_site:
                atom_mask = features["res_atom_mask"].bool().clone()
            else:
                atom_mask = (
                    features["res_atom_mask"].bool()
                    & features["binding_site_mask_clean"][:, None]
                )
        if pl_interface:
            # Removing ambiguous atoms
            atom_mask[:, [6, 7, 12, 13, 16, 17, 20, 21, 26, 27, 29, 30]] = False

        batch_size = batch["metadata"]["num_structid"]
        pred_prot_coords = pred_prot_coords[atom_mask].view(batch_size, -1, 3)
        if pred_lig_coords is not None:
            assert target_prot_coords is not None
            assert ligatm_types is not None
            pred_lig_coords = pred_lig_coords.view(batch_size, -1, 3)
            pred_coords = torch.cat([pred_prot_coords, pred_lig_coords], dim=1)
        else:
            pred_coords = pred_prot_coords
        sampling_rate = cap_size / pred_coords.shape[1]
        sampling_mask = (
            torch.rand(pred_coords.shape[1], device=self.device) < sampling_rate
        )
        pred_coords = pred_coords[:, sampling_mask]
        if pl_interface:
            pred_dist = (
                torch.square(pred_coords[:, :, None] - pred_lig_coords[:, None, :])
                .sum(-1)
                .add(1e-4)
                .sqrt()
                .sub(1e-2)
            )
        else:
            pred_dist = (
                torch.square(pred_coords[:, :, None] - pred_coords[:, None, :])
                .sum(-1)
                .add(1e-4)
                .sqrt()
                .sub(1e-2)
            )
        with torch.no_grad():
            target_prot_coords = target_prot_coords[atom_mask].view(batch_size, -1, 3)
            if pred_lig_coords is not None:
                target_lig_coords = target_lig_coords.view(batch_size, -1, 3)
                target_coords = torch.cat(
                    [target_prot_coords, target_lig_coords], dim=1
                )
            else:
                target_coords = target_prot_coords
            target_coords = target_coords[:, sampling_mask]
            if pl_interface:
                target_dist = (
                    torch.square(
                        target_coords[:, :, None] - target_lig_coords[:, None, :]
                    )
                    .sum(-1)
                    .add(1e-4)
                    .sqrt()
                    .sub(1e-2)
                )
            else:
                target_dist = (
                    torch.square(target_coords[:, :, None] - target_coords[:, None, :])
                    .sum(-1)
                    .add(1e-4)
                    .sqrt()
                    .sub(1e-2)
                )
            # In Angstrom, using UFF params to compute clash loss
            protatm_types = features["res_atom_types"].long()[atom_mask]
            protatm_vdw = self.atnum2vdw_uff[protatm_types].view(batch_size, -1)
            if pred_lig_coords is not None:
                ligatm_vdw = self.atnum2vdw_uff[ligatm_types].view(batch_size, -1)
                atm_vdw = torch.cat([protatm_vdw, ligatm_vdw], dim=1)
            else:
                atm_vdw = protatm_vdw
            atm_vdw = atm_vdw[:, sampling_mask]
            average_vdw = (atm_vdw[:, :, None] + atm_vdw[:, None, :]) / 2
            # Use conservative cutoffs to avoid mis-penalization

        dist_errors = (pred_dist - target_dist).square()
        drmsd = dist_errors.add(1e-2).sqrt().sub(1e-1).mean(dim=(1, 2))
        if pl_interface:
            return drmsd, None

        covalent_like = target_dist < (average_vdw * 1.2)
        # Alphafold supplementary Eq. 46, modified
        clash_pairwise = torch.clamp(average_vdw * 1.1 - pred_dist.add(1e-6), min=0.0)
        clash_loss = clash_pairwise.mul(~covalent_like).sum(dim=2).mean(dim=1)
        return drmsd, clash_loss

    def compute_per_atom_lddt(self, batch, pred_coords, target_coords):
        pred_coords = pred_coords.contiguous().view(
            batch["metadata"]["num_structid"], -1, 3
        )
        target_coords = target_coords.contiguous().view(
            batch["metadata"]["num_structid"], -1, 3
        )
        target_dist = (target_coords[:, :, None] - target_coords[:, None, :]).norm(
            dim=-1
        )
        pred_dist = (pred_coords[:, :, None] - pred_coords[:, None, :]).norm(dim=-1)
        conserved_mask = target_dist < 15.0
        lddt_list = []
        thresholds = [0, 0.5, 1, 2, 4, 6, 8, 12, 1e9]
        for threshold_idx in range(8):
            distdiff = (pred_dist - target_dist).abs()
            bin_fraction = (distdiff > thresholds[threshold_idx]) & (
                distdiff < thresholds[threshold_idx + 1]
            )
            lddt_list.append(
                bin_fraction.mul(conserved_mask).long().sum(dim=2)
                / conserved_mask.long().sum(dim=2)
            )
        lddt_list = torch.stack(lddt_list, dim=-1)
        lddt = torch.cumsum(lddt_list[:, :, :4], dim=-1).mean(dim=-1)
        return lddt, lddt_list

    def compute_lddt_ca(self, batch, pred_coords, target_coords):
        pred_coords = pred_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
        target_coords = target_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
        pred_ca_flat = pred_coords[:, :, 1]
        target_ca_flat = target_coords[:, :, 1]
        target_dist = (target_ca_flat[:, :, None] - target_ca_flat[:, None, :]).norm(
            dim=-1
        )
        pred_dist = (pred_ca_flat[:, :, None] - pred_ca_flat[:, None, :]).norm(dim=-1)
        conserved_mask = target_dist < 15.0
        lddt = 0
        for threshold in [0.5, 1, 2, 4]:
            below_threshold = (pred_dist - target_dist).abs() < threshold
            lddt = lddt + below_threshold.mul(conserved_mask).sum(
                (1, 2)
            ) / conserved_mask.sum((1, 2))
        return lddt / 4

    def compute_lddt_pli(
        self,
        batch,
        pred_prot_coords,
        target_prot_coords,
        pred_lig_coords,
        target_lig_coords,
    ):
        features = batch["features"]
        batch_size = batch["metadata"]["num_structid"]
        atom_mask = features["res_atom_mask"].bool()
        pred_prot_coords = pred_prot_coords[atom_mask].view(batch_size, -1, 3)
        target_prot_coords = target_prot_coords[atom_mask].view(batch_size, -1, 3)
        pred_lig_coords = pred_lig_coords.view(batch_size, -1, 3)
        target_lig_coords = target_lig_coords.view(batch_size, -1, 3)
        target_dist = (
            target_prot_coords[:, :, None] - target_lig_coords[:, None, :]
        ).norm(dim=-1)
        pred_dist = (pred_prot_coords[:, :, None] - pred_lig_coords[:, None, :]).norm(
            dim=-1
        )
        conserved_mask = target_dist < 6.0
        lddt = 0
        for threshold in [0.5, 1, 2, 4]:
            below_threshold = (pred_dist - target_dist).abs() < threshold
            lddt = lddt + below_threshold.mul(conserved_mask).sum(
                (1, 2)
            ) / conserved_mask.sum((1, 2))
        return lddt / 4

    def compute_TMscore_raw(self, batch, pred_coords, target_coords):
        pred_coords = pred_coords.view(batch["metadata"]["num_structid"], -1, 3)
        target_coords = target_coords.view(batch["metadata"]["num_structid"], -1, 3)
        pair_dist_aligned = (pred_coords - target_coords).norm(dim=-1)
        tm_normalizer = 1.24 * (max(target_coords.shape[1], 19) - 15) ** (1 / 3) - 1.8
        per_struct_tm = torch.mean(
            1 / (1 + (pair_dist_aligned / tm_normalizer) ** 2), dim=1
        )
        return per_struct_tm

    def compute_TMscore_lbound(self, batch, pred_coords, target_coords):
        features = batch["features"]
        atom_mask = (
            features["res_atom_mask"]
            .bool()
            .view(batch["metadata"]["num_structid"], -1, 37)
        )
        pred_coords = pred_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
        target_coords = target_coords.view(batch["metadata"]["num_structid"], -1, 37, 3)
        pred_bb_frames = get_frame_matrix(
            pred_coords[:, :, 0, :],
            pred_coords[:, :, 1, :],
            pred_coords[:, :, 2, :],
            strict=True,
        )
        target_bb_frames = get_frame_matrix(
            target_coords[:, :, 0, :],
            target_coords[:, :, 1, :],
            target_coords[:, :, 2, :],
            strict=True,
        )
        pred_coords_flat = pred_coords[atom_mask].view(
            batch["metadata"]["num_structid"], -1, 3
        )
        target_coords_flat = target_coords[atom_mask].view(
            batch["metadata"]["num_structid"], -1, 3
        )
        # Columns-frames, rows-points
        # [B, 1, N, 3] - [B, F, 1, 3]
        aligned_pred_points = cartesian_to_internal(
            pred_coords_flat.unsqueeze(1), pred_bb_frames.unsqueeze(2)
        )
        with torch.no_grad():
            aligned_target_points = cartesian_to_internal(
                target_coords_flat.unsqueeze(1), target_bb_frames.unsqueeze(2)
            )
        pair_dist_aligned = (aligned_pred_points - aligned_target_points).norm(dim=-1)
        tm_normalizer = 1.24 * (max(target_coords.shape[1], 19) - 15) ** (1 / 3) - 1.8
        per_frame_tm = torch.mean(
            1 / (1 + (pair_dist_aligned / tm_normalizer) ** 2), dim=2
        )
        return torch.amax(per_frame_tm, dim=1)

    def compute_fape_from_atom37(
        self,
        batch,
        pred_prot_coords,  # [N_res, 37, 3]
        target_prot_coords,  # [N_res, 37, 3]
        pred_lig_coords=None,  # [N_atom, 3]
        target_lig_coords=None,  # [N_atom, 3]
        lig_frame_atm_idx=None,  # [3, N_atom]
        split_pl_views=False,
        cap_size=8000,
    ):
        features = batch["features"]
        batch_size = batch["metadata"]["num_structid"]
        with torch.no_grad():
            atom_mask = (
                features["res_atom_mask"]
                .bool()
                .view(batch["metadata"]["num_structid"], -1, 37)
            ).clone()
            atom_mask[:, :, [6, 7, 12, 13, 16, 17, 20, 21, 26, 27, 29, 30]] = False
        pred_prot_coords = pred_prot_coords.view(batch_size, -1, 37, 3)
        target_prot_coords = target_prot_coords.view(batch_size, -1, 37, 3)
        pred_bb_frames = get_frame_matrix(
            pred_prot_coords[:, :, 0, :],
            pred_prot_coords[:, :, 1, :],
            pred_prot_coords[:, :, 2, :],
        )
        # pred_bb_frames.R = pred_bb_frames.R.detach()
        target_bb_frames = get_frame_matrix(
            target_prot_coords[:, :, 0, :],
            target_prot_coords[:, :, 1, :],
            target_prot_coords[:, :, 2, :],
        )
        pred_prot_coords_flat = pred_prot_coords[atom_mask].view(batch_size, -1, 3)
        target_prot_coords_flat = target_prot_coords[atom_mask].view(batch_size, -1, 3)
        if pred_lig_coords is not None:
            assert target_prot_coords is not None
            assert lig_frame_atm_idx is not None
            pred_lig_coords = pred_lig_coords.view(batch_size, -1, 3)
            target_lig_coords = target_lig_coords.view(batch_size, -1, 3)
            pred_coords = torch.cat([pred_prot_coords_flat, pred_lig_coords], dim=1)
            target_coords = torch.cat(
                [target_prot_coords_flat, target_lig_coords], dim=1
            )
            pred_lig_frames = get_frame_matrix(
                pred_lig_coords[:, lig_frame_atm_idx[0]],
                pred_lig_coords[:, lig_frame_atm_idx[1]],
                pred_lig_coords[:, lig_frame_atm_idx[2]],
            )
            pred_frames = pred_bb_frames.concatenate(pred_lig_frames, dim=1)
            target_lig_frames = get_frame_matrix(
                target_lig_coords[:, lig_frame_atm_idx[0]],
                target_lig_coords[:, lig_frame_atm_idx[1]],
                target_lig_coords[:, lig_frame_atm_idx[2]],
            )
            target_frames = target_bb_frames.concatenate(target_lig_frames, dim=1)
        else:
            pred_coords = pred_prot_coords_flat
            target_coords = target_prot_coords_flat
            pred_frames = pred_bb_frames
            target_frames = target_bb_frames
        # Columns-frames, rows-points
        # [B, 1, N, 3] - [B, F, 1, 3]
        sampling_rate = cap_size / (batch_size * target_coords.shape[1])
        sampling_mask = (
            torch.rand(target_coords.shape[1], device=self.device) < sampling_rate
        )
        aligned_pred_points = cartesian_to_internal(
            pred_coords[:, sampling_mask].unsqueeze(1), pred_frames.unsqueeze(2)
        )
        with torch.no_grad():
            aligned_target_points = cartesian_to_internal(
                target_coords[:, sampling_mask].unsqueeze(1), target_frames.unsqueeze(2)
            )
        pair_dist_aligned = (
            torch.square(aligned_pred_points - aligned_target_points)
            .sum(-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
        )
        cropped_pair_dists = torch.clamp(pair_dist_aligned, max=10)
        normalized_pair_dists = (
            pair_dist_aligned / aligned_target_points.square().sum(-1).add(1e-4).sqrt()
        )
        if split_pl_views:
            fape_protframe = (
                cropped_pair_dists[:, : target_bb_frames.t.shape[1]].mean((1, 2)) / 10
            )
            fape_ligframe = (
                cropped_pair_dists[:, target_bb_frames.t.shape[1] :].mean((1, 2)) / 10
            )
            return fape_protframe, fape_ligframe, normalized_pair_dists.mean((1, 2))
        return cropped_pair_dists.mean((1, 2)) / 10, normalized_pair_dists.mean((1, 2))

    def compute_protein_distogram_loss(
        self, batch, target_coords, entry="res_res_grid_attr_flat"
    ):
        n_protein_patches = batch["metadata"]["n_prot_patches_per_sample"]
        sampled_grid_features = batch["features"][entry]
        sampled_ca_coords = target_coords[batch["indexer"]["gather_idx_pid_a"]].view(
            batch["metadata"]["num_structid"], n_protein_patches, 3
        )
        sampled_ca_dist = torch.norm(
            sampled_ca_coords[:, :, None] - sampled_ca_coords[:, None, :], dim=-1
        )
        # Using AF2 parameters
        distance_bin_idx = torch.bucketize(
            sampled_ca_dist, self.dist_bins[:-1], right=True
        )
        distogram_loss = F.cross_entropy(
            self.dgram_head(sampled_grid_features), distance_bin_idx.flatten()
        )
        return distogram_loss

    def compute_aa_distance_geometry_loss(self, batch, pred_coords, target_coords):
        batch_size = batch["metadata"]["num_structid"]
        features = batch["features"]
        atom_mask = features["res_atom_mask"].bool()
        # Add backbone atoms from previous residue
        atom_mask = atom_mask.view(batch_size, -1, 37)
        atom_mask = torch.cat(
            [atom_mask[:, 1:], atom_mask[:, :-1, 0:3]], dim=2
        ).flatten(0, 1)
        pred_coords = pred_coords.view(batch_size, -1, 37, 3)
        pred_coords = torch.cat(
            [pred_coords[:, 1:], pred_coords[:, :-1, 0:3]], dim=2
        ).flatten(0, 1)
        target_coords = target_coords.view(batch_size, -1, 37, 3)
        target_coords = torch.cat(
            [target_coords[:, 1:], target_coords[:, :-1, 0:3]], dim=2
        ).flatten(0, 1)
        local_pair_dist_target = (
            (target_coords[:, None, :] - target_coords[:, :, None])
            .square()
            .sum(-1)
            .add(1e-4)
            .sqrt()
        )
        local_pair_dist_pred = (
            (pred_coords[:, None, :] - pred_coords[:, :, None])
            .square()
            .sum(-1)
            .add(1e-4)
            .sqrt()
        )
        local_pair_mask = (
            atom_mask[:, None, :]
            & atom_mask[:, :, None]
            & (local_pair_dist_target < 3.0)
        )
        ret = (local_pair_dist_target - local_pair_dist_pred).abs()[local_pair_mask]
        return ret.view(batch["metadata"]["num_structid"], -1).mean(dim=1)

    def compute_sm_distance_geometry_loss(self, batch, pred_coords, target_coords):
        batch_size = batch["metadata"]["num_structid"]
        pred_coords = pred_coords.view(batch_size, -1, 3)
        target_coords = target_coords.view(batch_size, -1, 3)
        pair_dist_target = (
            (target_coords[:, None, :] - target_coords[:, :, None])
            .square()
            .sum(-1)
            .add(1e-4)
            .sqrt()
        )
        pair_dist_pred = (
            (pred_coords[:, None, :] - pred_coords[:, :, None])
            .square()
            .sum(-1)
            .add(1e-4)
            .sqrt()
        )
        local_pair_mask = pair_dist_target < 3.0
        ret = (pair_dist_target - pair_dist_pred).abs()[local_pair_mask]
        return ret.view(batch_size, -1).mean(dim=1)

    def _distance_to_gaussian_contact_logits(self, x, lower_bound=-16.0, cutoff=None):
        if cutoff is None:
            cutoff = self.CONTACT_SCALE * 2
        return torch.log(torch.clamp(1 - (x / cutoff), min=1e-9))

    def _distogram_to_gaussian_contact_logits(self, dgram):
        return torch.logsumexp(
            dgram + self._distance_to_gaussian_contact_logits(self.dist_bins),
            dim=-1,
        )

    def _softmax_contact_ca_transform(
        self, batch, L, x, block_mask=None, return_masked_L=False
    ):
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = batch["metadata"]["num_structid"]
        n_a_per_sample = max(metadata["num_a_per_sample"])
        max(metadata["num_i_per_sample"])
        n_A_per_sample = metadata["n_prot_patches_per_sample"]
        n_I_per_sample = metadata["n_lig_patches_per_sample"]
        L = L.view(batch_size, n_A_per_sample, n_I_per_sample)
        if block_mask is not None:
            gathered_block_mask = (
                block_mask.transpose(1, 2)
                .contiguous()
                .flatten(0, 1)[indexer["gather_idx_I_molid"]]
                .contiguous()
                .flatten(batch_size, n_I_per_sample, n_A_per_sample)
                .transpose(1, 2)
                .contiguous()
            )
            L = L - (~gathered_block_mask) * 1e9
        outs = []
        x = x[indexer["gather_idx_pid_a"]].view(batch_size, n_A_per_sample, 3)
        for ligand_id in range(max(metadata["num_molid_per_sample"])):
            lig_id_mask = indexer["gather_idx_I_molid"] == ligand_id
            c_lig = torch.softmax(L[:, :, lig_id_mask].flatten(1, 2), dim=1).view(
                batch_size, n_a_per_sample, -1
            )
            outs.append(torch.bmm(c_lig.transpose(1, 2), x).sum(dim=1, keepdim=True))
        out = torch.cat(outs, dim=1).flatten(0, 1)
        if return_masked_L:
            return out, L
        return out

    def _eval_true_contact_maps(self, batch, **kwargs):
        indexer = batch["indexer"]
        batch_size = batch["metadata"]["num_structid"]
        with torch.no_grad():
            # Residue centroids
            res_cent_coords = (
                batch["features"]["res_atom_positions"]
                .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
                .sum(dim=1)
                .div(
                    batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None] + 1e-9
                )
            )
            res_lig_dist = (
                res_cent_coords.view(batch_size, -1, 3)[:, :, None]
                - batch["features"]["sdf_coordinates"][indexer["gather_idx_U_u"]].view(
                    batch_size, -1, 3
                )[:, None, :]
            ).norm(dim=-1)
            res_lig_contact_logit = self._distance_to_gaussian_contact_logits(
                res_lig_dist, **kwargs
            )
        return res_lig_dist, res_lig_contact_logit.flatten()

    def compute_contact_prediction_losses(
        self,
        batch,
        pred_distograms: torch.Tensor,
        ref_dist_mat: torch.Tensor,
    ):
        # True onehot distance and distogram loss
        distance_bin_idx = torch.bucketize(
            ref_dist_mat, self.dist_bins[:-1], right=True
        )
        distogram_loss = F.cross_entropy(
            pred_distograms.flatten(0, -2), distance_bin_idx.flatten()
        )
        # Evaluate contact logits via log(\sum_j p_j \exp(-\alpha*r_j^2))
        ref_contact_logits = self._distance_to_gaussian_contact_logits(ref_dist_mat)
        pred_contact_logits = self._distogram_to_gaussian_contact_logits(
            pred_distograms
        )
        forward_kl_loss = F.kl_div(
            F.log_softmax(
                pred_contact_logits.flatten(-2, -1),
                dim=-1,
            ),
            F.log_softmax(
                ref_contact_logits.flatten(-2, -1),
                dim=-1,
            ),
            log_target=True,
            reduction="batchmean",
        )
        return distogram_loss, forward_kl_loss

    def compute_fape_loss(
        self, batch, struct, target_struct, atom_weight=None, average=True
    ):
        indexer = batch["indexer"]
        features = batch["features"]
        atom_mask = ~features["target_res_atom_mask"].bool()
        frames, local_coords = struct
        src_idx, dst_idx = indexer["gather_idx_ab_a"], indexer["gather_idx_ab_b"]
        points_src = local_coords[src_idx]
        points_dst = cartesian_to_internal(
            internal_to_cartesian(points_src, frames[src_idx]),
            frames[dst_idx],
        )
        target_frames, target_local_coords = target_struct
        target_points_src = target_local_coords[src_idx]
        target_points_dst = cartesian_to_internal(
            internal_to_cartesian(target_points_src, target_frames[src_idx]),
            target_frames[dst_idx],
        )
        atom_mask_src = atom_mask[src_idx]
        pair_dist = torch.square(points_dst - target_points_dst).sum(-1).add(1e-4)
        if atom_weight is None:
            weight = torch.ones_like(pair_dist).mul(~atom_mask_src)
        else:
            weight = atom_weight[src_idx].mul(~atom_mask_src)
        pair_dist = pair_dist * weight
        if not average:
            return segment_sum(
                pair_dist.mul(weight).sum(-1).sqrt(), src_idx, len(atom_mask)
            ) / segment_sum(weight.sum(-1).sqrt(), src_idx, len(atom_mask))
        return pair_dist.mul(weight).sum().div(weight.sum() + 1e-4).sqrt()

    def _prepare_protein_patch_indexers(self, batch, randomize_anchors=False):
        metadata = batch["metadata"]
        indexer = batch["indexer"]
        batch_size = metadata["num_structid"]
        # Prepare indexers
        # Use max to ensure segmentation faults are 100% invoked
        # in case there are any bad indices
        n_a_per_sample = max(metadata["num_a_per_sample"])
        assert n_a_per_sample * batch_size == metadata["num_a"]
        n_protein_patches = min(self.protein_config.n_patches, n_a_per_sample)
        batch["metadata"]["n_prot_patches_per_sample"] = n_protein_patches

        # Uniform segmentation
        res_idx_in_batch = torch.arange(metadata["num_a"], device=self.device)
        batch["indexer"]["gather_idx_a_pid"] = (
            res_idx_in_batch // n_a_per_sample
        ) * n_protein_patches + (
            ((res_idx_in_batch % n_a_per_sample) * n_protein_patches) // n_a_per_sample
        )

        if randomize_anchors:
            # Random down-sampling, assigning residues to the patch grid
            # This maps grid row/column idx to sampled residue idx
            batch["indexer"]["gather_idx_pid_a"] = segment_argmin(
                batch["features"]["res_type"].new_zeros(n_a_per_sample * batch_size),
                indexer["gather_idx_a_pid"],
                n_protein_patches * batch_size,
                randomize=True,
            )
        else:
            batch["indexer"]["gather_idx_pid_a"] = segment_mean(
                res_idx_in_batch,
                indexer["gather_idx_a_pid"],
                n_protein_patches * batch_size,
            ).long()

        return batch

    def _prepare_protein_backbone_indexers(self, batch, **kwargs):
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]

        protatm_coords_padded = features["input_protein_coords"]
        batch_size = metadata["num_structid"]

        assert self.global_config.single_protein_batch
        num_res_per_struct = max(metadata["num_a_per_sample"])
        # Check that the samples are clones of the same complex
        assert batch_size * num_res_per_struct == protatm_coords_padded.shape[0]

        input_prot_coords_folded = protatm_coords_padded.unflatten(
            0, (batch_size, num_res_per_struct)
        )
        single_struct_chain_id = indexer["gather_idx_a_chainid"][:num_res_per_struct]
        single_struct_res_id = features["residue_index"][:num_res_per_struct]
        ca_ca_dist = (
            input_prot_coords_folded[:, :, None, 1]
            - input_prot_coords_folded[:, None, :, 1]
        ).norm(dim=-1)
        ca_ca_knn_mask = topk_edge_mask_from_logits(
            -ca_ca_dist / self.CONTACT_SCALE,
            self.protein_config.max_residue_degree,
            randomize=True,
        )
        chain_mask = (
            single_struct_chain_id[None, :, None]
            == single_struct_chain_id[None, None, :]
        )
        sequence_dist = (
            single_struct_res_id[None, :, None] - single_struct_res_id[None, None, :]
        )
        sequence_proximity_mask = (torch.abs(sequence_dist) <= 4) & chain_mask
        prot_res_res_edge_mask = ca_ca_knn_mask | sequence_proximity_mask

        dense_row_idx_3D = (
            torch.arange(batch_size * num_res_per_struct, device=self.device)
            .view(batch_size, num_res_per_struct)[:, :, None]
            .expand(-1, -1, num_res_per_struct)
        ).contiguous()
        dense_col_idx_3D = dense_row_idx_3D.transpose(1, 2).contiguous()
        batch["metadata"]["num_prot_res"] = metadata["num_a"]
        batch["indexer"]["gather_idx_ab_a"] = dense_row_idx_3D[prot_res_res_edge_mask]
        batch["indexer"]["gather_idx_ab_b"] = dense_col_idx_3D[prot_res_res_edge_mask]
        batch["indexer"]["gather_idx_ab_structid"] = indexer["gather_idx_a_structid"][
            indexer["gather_idx_ab_a"]
        ]
        batch["metadata"]["num_ab"] = batch["indexer"]["gather_idx_ab_a"].shape[0]

        if self.global_config.constrained_inpainting:
            # Diversified spherical cropping scheme
            assert self.global_config.single_protein_batch
            batch_size = metadata["num_structid"]
            # Assert single ligand samples
            assert batch_size == metadata["num_molid"]
            ligand_coords = batch["features"]["sdf_coordinates"].reshape(
                batch_size, -1, 3
            )
            ligand_centroids = torch.mean(ligand_coords, dim=1)
            if self.training:
                # 3A perturbations around the ligand centroid
                perturbed_centroids = (
                    ligand_centroids + torch.rand_like(ligand_centroids) * 1.73
                )
                site_radius = torch.amax(
                    torch.norm(ligand_coords - perturbed_centroids[:, None, :], dim=-1),
                    dim=1,
                )
                perturbed_site_radius = (
                    site_radius
                    + (0.5 + torch.rand_like(site_radius)) * self.BINDING_SITE_CUTOFF
                )
            else:
                perturbed_centroids = ligand_centroids
                site_radius = torch.amax(
                    torch.norm(ligand_coords - perturbed_centroids[:, None, :], dim=-1),
                    dim=1,
                )
                perturbed_site_radius = site_radius + self.BINDING_SITE_CUTOFF
            centroid_ca_dist = (
                batch["features"]["res_atom_positions"][:, 1]
                .contiguous()
                .view(batch_size, -1, 3)
                - perturbed_centroids[:, None, :]
            ).norm(dim=-1)
            binding_site_mask = (
                centroid_ca_dist < perturbed_site_radius[:, None]
            ).flatten(0, 1)
            batch["features"]["binding_site_mask"] = binding_site_mask
            batch["features"]["template_alignment_mask"] = (~binding_site_mask) & batch[
                "features"
            ]["template_alignment_mask"].bool()

        return batch

    def _initialize_protatm_indexer_and_embeddings(self, batch):
        """Assign coordinate-independent edges and protein features from PIFormer"""
        assert self.global_config.single_protein_batch
        self._standard_aa_molgraph_featset = inplace_to_device(
            self._standard_aa_molgraph_featset, self.device
        )
        self._standard_aa_template_featset = inplace_to_device(
            self._standard_aa_template_featset, self.device
        )
        self._standard_aa_molgraph_featset = self.ligand_encoder(
            self._standard_aa_molgraph_featset
        )
        with torch.no_grad():
            assert self._standard_aa_molgraph_featset["metadata"]["num_i"] == 167
            template_atom_idx_in_batch_padded = torch.full(
                (20, 37), fill_value=-1, dtype=torch.long, device=self.device
            )
            template_atom37_mask = self._standard_aa_template_featset["features"][
                "res_atom_mask"
            ].bool()
            template_atom_idx_in_batch_padded[template_atom37_mask] = torch.arange(
                167, device=self.device
            )
            atom_idx_in_batch_to_restype_idx = (
                torch.arange(20, device=self.device)[:, None]
                .expand(-1, 37)
                .contiguous()[template_atom37_mask]
            )
            atom_idx_in_batch_to_atom37_idx = (
                torch.arange(37, device=self.device)[None, :]
                .expand(20, -1)
                .contiguous()[template_atom37_mask]
            )
            template_padded_edge_mask_per_aa = torch.zeros(
                (20, 37, 37), dtype=torch.bool, device=self.device
            )
            template_aa_graph_indexer = self._standard_aa_molgraph_featset["indexer"]
            template_padded_edge_mask_per_aa[
                atom_idx_in_batch_to_restype_idx[
                    template_aa_graph_indexer["gather_idx_uv_u"]
                ],
                atom_idx_in_batch_to_atom37_idx[
                    template_aa_graph_indexer["gather_idx_uv_u"]
                ],
                atom_idx_in_batch_to_atom37_idx[
                    template_aa_graph_indexer["gather_idx_uv_v"]
                ],
            ] = True

            # Gather adjacency matrix to the input protein
            features = batch["features"]
            metadata = batch["metadata"]
            # Prepare intra-residue protein atom - protein atom indexers
            n_res_first = max(metadata["num_a_per_sample"])
            batch["features"]["res_atom_mask"] = features["res_atom_mask"].bool()
            protatm_padding_mask = batch["features"]["res_atom_mask"][:n_res_first]
            n_protatm_first = int(protatm_padding_mask.sum())
            protatm_res_idx_res_first = (
                torch.arange(n_res_first, device=self.device)[:, None]
                .expand(-1, 37)
                .contiguous()[protatm_padding_mask]
            )
            protatm_to_atom37_idx_first = (
                torch.arange(37, device=self.device)[None, :]
                .expand(n_res_first, -1)
                .contiguous()[protatm_padding_mask]
            )
            same_residue_mask = (
                protatm_res_idx_res_first[:, None] == protatm_res_idx_res_first[None, :]
            ).contiguous()
            aa_graph_edge_mask = torch.zeros(
                (n_protatm_first, n_protatm_first), dtype=torch.bool, device=self.device
            )
            src_idx_sameres = (
                torch.arange(n_protatm_first, device=self.device)[:, None]
                .expand(-1, n_protatm_first)
                .contiguous()[same_residue_mask]
            )
            dst_idx_sameres = (
                torch.arange(n_protatm_first, device=self.device)[None, :]
                .expand(n_protatm_first, -1)
                .contiguous()[same_residue_mask]
            )
            aa_graph_edge_mask[
                src_idx_sameres, dst_idx_sameres
            ] = template_padded_edge_mask_per_aa[
                features["res_type"].long()[protatm_res_idx_res_first[src_idx_sameres]],
                protatm_to_atom37_idx_first[src_idx_sameres],
                protatm_to_atom37_idx_first[dst_idx_sameres],
            ]
            src_idx_first = (
                torch.arange(n_protatm_first, device=self.device)[:, None]
                .expand(-1, n_protatm_first)
                .contiguous()[aa_graph_edge_mask]
            )
            dst_idx_first = (
                torch.arange(n_protatm_first, device=self.device)[None, :]
                .expand(n_protatm_first, -1)
                .contiguous()[aa_graph_edge_mask]
            )
            batch_size = metadata["num_structid"]
            src_idx = (
                (
                    src_idx_first[None, :].expand(batch_size, -1)
                    + torch.arange(batch_size, device=self.device)[:, None]
                    * n_protatm_first
                )
                .contiguous()
                .flatten()
            )
            dst_idx = (
                (
                    dst_idx_first[None, :].expand(batch_size, -1)
                    + torch.arange(batch_size, device=self.device)[:, None]
                    * n_protatm_first
                )
                .contiguous()
                .flatten()
            )
            batch["metadata"]["num_protatm_per_sample"] = n_protatm_first
            batch["indexer"]["protatm_protatm_idx_src"] = src_idx
            batch["indexer"]["protatm_protatm_idx_dst"] = dst_idx
            batch["metadata"]["num_prot_atm"] = n_protatm_first * batch_size
            batch["indexer"]["protatm_res_idx_res"] = (
                (
                    protatm_res_idx_res_first[None, :].expand(batch_size, -1)
                    + torch.arange(batch_size, device=self.device)[:, None]
                    * n_res_first
                )
                .contiguous()
                .flatten()
            )
            batch["indexer"]["protatm_res_idx_protatm"] = torch.arange(
                batch["metadata"]["num_prot_atm"], device=self.device
            )
        # Gather graph features to the protein feature set
        template_padded_node_feat_per_aa = torch.zeros(
            (20, 37, self.protein_config.residue_dim), device=self.device
        )
        template_padded_node_feat_per_aa[
            template_atom37_mask
        ] = self.molgraph_single_projector(
            self._standard_aa_molgraph_featset["features"]["lig_atom_attr"]
        )
        protatm_padding_mask = batch["features"]["res_atom_mask"]
        protatm_to_atom37_idx = (
            protatm_to_atom37_idx_first[None, :]
            .expand(batch_size, -1)
            .contiguous()
            .flatten(0, 1)
        )
        batch["features"]["protatm_to_atom37_index"] = protatm_to_atom37_idx
        batch["features"]["protatm_to_atomic_number"] = features[
            "res_atom_types"
        ].long()[protatm_padding_mask]
        batch["features"]["prot_atom_attr_projected"] = (
            template_padded_node_feat_per_aa[
                features["res_type"].long()[protatm_res_idx_res_first],
                protatm_to_atom37_idx_first,
            ][None, :]
            .expand(batch_size, -1, -1)
            .contiguous()
            .flatten(0, 1)
        )
        template_padded_edge_feat_per_aa = torch.zeros(
            (20, 37, 37, self.protein_config.pair_dim), device=self.device
        )
        template_padded_edge_feat_per_aa[
            atom_idx_in_batch_to_restype_idx[
                template_aa_graph_indexer["gather_idx_uv_u"]
            ],
            atom_idx_in_batch_to_atom37_idx[
                template_aa_graph_indexer["gather_idx_uv_u"]
            ],
            atom_idx_in_batch_to_atom37_idx[
                template_aa_graph_indexer["gather_idx_uv_v"]
            ],
        ] = self.molgraph_pair_projector(
            self._standard_aa_molgraph_featset["features"]["lig_atom_pair_attr"]
        )

        batch["features"]["prot_atom_pair_attr_projected"] = (
            template_padded_edge_feat_per_aa[
                features["res_type"].long()[protatm_res_idx_res_first[src_idx_first]],
                protatm_to_atom37_idx_first[src_idx_first],
                protatm_to_atom37_idx_first[dst_idx_first],
            ][None, :, :]
            .expand(batch_size, -1, -1)
            .contiguous()
            .flatten(0, 1)
        )
        return batch

    def step(self, batch, batch_idx, stage, time_ratio=None, max_chain_dist_k=None):
        if "outputs" not in batch.keys():
            batch["outputs"] = {}

        if self.global_config.task_type == "LBA":
            out = self.forward(batch)
            labels = batch["labels"]
            loss = F.mse_loss(out, labels)
            self.log(
                f"{stage}_loss", loss, on_epoch=True, batch_size=batch["batch_size"]
            )
            return {"loss": loss, "out": out, "labels": labels}
        elif self.global_config.task_type == "all_atom_prediction":
            # if batch_idx % 2 == 0:
            #     return self._eval_confidence_estimation_losses(
            #         batch, batch_idx, self.device, stage
            #     )
            return self._eval_structure_prediction_losses(
                batch,
                batch_idx,
                self.device,
                stage,
                T=1000,
                time_ratio=time_ratio,
            )

    def _resolve_latent_converter(self, *args):
        if self.config.latent_model == "default":
            return DefaultPLCoordinateConverter(self.global_config, *args)
        elif self.config.latent_model == "branched_polymer":
            return BranchedGaussianChainConverter(self.global_config, *args)
        elif self.config.latent_model == "null":
            return NullPLCoordinateConverter(self.global_config, *args)
        else:
            raise NotImplementedError

    def _eval_structure_prediction_losses(
        self, batch, batch_idx, device, stage, T=1000, time_ratio=None
    ):
        if (
            "num_molid" in batch["metadata"].keys()
            and batch["metadata"]["num_molid"] > 0
        ):
            batch["misc"]["protein_only"] = False
        else:
            batch["misc"]["protein_only"] = True

        if "augmented_coordinates" in batch["features"].keys():
            batch["features"]["sdf_coordinates"] = batch["features"][
                "augmented_coordinates"
            ]
            is_native_sample = 0
        else:
            is_native_sample = 1

        # Sample the timestep for each structure
        if time_ratio is not None:
            t = int(time_ratio * T)
            t = torch.full((batch["metadata"]["num_structid"],), t, device=device)[
                :, None
            ]
        else:
            # importance_sampling = bool(random.randint(0, 1))
            importance_sampling = False
            if importance_sampling:
                t = (
                    torch.log(
                        1
                        + torch.rand(
                            (batch["metadata"]["num_structid"],), device=device
                        )[:, None]
                        * 1000
                    )
                    * T
                    / math.log(1000)
                )
            else:
                t = (
                    torch.rand((batch["metadata"]["num_structid"],), device=device)[
                        :, None
                    ]
                    * T
                )

        prior_training = int(random.randint(0, 10) == 1)
        if prior_training == 1:
            t = torch.full_like(t, T)

        if self.training and self.global_config.use_template:
            use_template = bool(random.randint(0, 1))
        else:
            use_template = self.global_config.use_template

        self._assign_timestep_encodings(batch, t / T)
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]

        loss = 0
        forward_lat_converter = self._resolve_latent_converter(
            [
                ("features", "res_atom_positions"),
                ("features", "input_protein_coords"),
            ],
            [("features", "sdf_coordinates"), ("features", "input_ligand_coords")],
        )
        batch_size = metadata["num_structid"]
        max(metadata["num_a_per_sample"])
        batch = self._prepare_protein_patch_indexers(batch)
        if batch["misc"]["protein_only"] == False:
            max(metadata["num_i_per_sample"])

            # Evaluate the contact map
            ref_dist_mat, contact_logit_matrix = self._eval_true_contact_maps(batch)
            num_cont_to_sample = max(metadata["num_I_per_sample"])
            sampled_block_contacts = [
                None,
            ]
            # Onehot contact code sampling
            with torch.no_grad():
                for _ in range(num_cont_to_sample):
                    sampled_block_contacts.append(
                        self._sample_reslig_contact_matrix(
                            batch, contact_logit_matrix, last=sampled_block_contacts[-1]
                        )
                    )
            forward_lat_converter.lig_res_anchor_mask = (
                self._sample_res_rowmask_from_contacts(batch, contact_logit_matrix)
            )
            with torch.no_grad():
                batch = self._forward_diffuse_plcomplex_latinp(
                    batch, t[:, :, None], T, forward_lat_converter
                )
            if prior_training == 1:
                iter_id = random.randint(0, num_cont_to_sample)
            else:
                iter_id = num_cont_to_sample
            # iter_id = num_cont_to_sample
            batch = self.forward(
                batch, contact_prediction=False, score=False, use_template=use_template
            )
            batch = self._run_contact_map_stack(
                batch,
                iter_id=iter_id,
                observed_block_contacts=sampled_block_contacts[iter_id],
            )
            pred_distogram = batch["outputs"][f"res_lig_distogram_out_{iter_id}"]
            (
                pl_distogram_loss,
                pl_contact_loss_forward,
            ) = self.compute_contact_prediction_losses(
                batch, pred_distogram, ref_dist_mat
            )
            cont_loss = 0
            cont_loss = (
                cont_loss
                + pl_distogram_loss
                * self.global_config.contact_loss_weight
                * is_native_sample
            )
            cont_loss = (
                cont_loss
                + pl_contact_loss_forward
                * self.global_config.contact_loss_weight
                * is_native_sample
            )
            self.log(
                f"{stage}_contact/contact_loss_distogram",
                pl_distogram_loss.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_contact/contact_loss_forwardKL",
                pl_contact_loss_forward.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        else:
            with torch.no_grad():
                batch = self._forward_diffuse_plcomplex_latinp(
                    batch, t[:, :, None], T, forward_lat_converter
                )
            iter_id = 0
            batch = self.forward(
                batch,
                iter_id=0,
                contact_prediction=True,
                score=False,
                use_template=use_template,
            )
        protein_distogram_loss = self.compute_protein_distogram_loss(
            batch,
            batch["features"]["res_atom_positions"][:, 1],
            entry=f"res_res_grid_attr_flat_out_{iter_id}",
        )
        self.log(
            f"{stage}_contact/prot_distogram_loss",
            protein_distogram_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )

        alpha_t, alpha_tm, beta_t = self._get_noise_schedule_vp(
            t + 1, T, decay_rate=self.global_config.sde_decay_rate * 2.5
        )
        # lambda_weighting = ((alpha_t * (t/T)).sqrt() / (1 - alpha_t + 1E-2)).squeeze(-1)
        # Weighting function from https://arxiv.org/pdf/2206.00364.pdf (TMTS objective)
        sigma_t = (
            (1 - alpha_t) / (alpha_t).sqrt()
        ) * 2 + 0.1  # * self.max_rescaled_sigma
        lambda_weighting = (
            sigma_t.square().squeeze(-1) + self.max_rescaled_sigma**2
        ).sqrt() / (sigma_t.squeeze(-1) * self.max_rescaled_sigma)
        # Run score head and evaluate structure prediction losses
        res_atom_mask = features["res_atom_mask"].bool()

        scores = self._run_score_head(batch, embedding_iter_id=iter_id)

        if self.training:
            # # Sigmoid scaling
            # violation_loss_ratio = 1 / (
            #     1
            #     + math.exp(10 - 12 * self.current_epoch / self.global_config.max_epoch)
            # )
            # violation_loss_ratio = (self.current_epoch / self.global_config.max_epoch)
            violation_loss_ratio = 1.0
        else:
            violation_loss_ratio = 1.0

        if not batch["misc"]["protein_only"]:
            if "binding_site_mask_clean" not in batch["features"]:
                with torch.no_grad():
                    min_lig_res_dist_clean = (
                        (
                            batch["features"]["res_atom_positions"][:, 1].view(
                                batch_size, -1, 3
                            )[:, :, None]
                            - batch["features"]["sdf_coordinates"].view(
                                batch_size, -1, 3
                            )[:, None, :]
                        )
                        .norm(dim=-1)
                        .amin(dim=2)
                    ).flatten(0, 1)
                    binding_site_mask_clean = (
                        min_lig_res_dist_clean < self.BINDING_SITE_CUTOFF
                    )
                batch["features"]["binding_site_mask_clean"] = binding_site_mask_clean
            coords_pred_prot = scores["final_coords_prot_atom_padded"][
                res_atom_mask
            ].view(metadata["num_structid"], -1, 3)
            coords_ref_prot = batch["features"]["res_atom_positions"][
                res_atom_mask
            ].view(metadata["num_structid"], -1, 3)
            coords_pred_bs_prot = scores["final_coords_prot_atom_padded"][
                res_atom_mask & batch["features"]["binding_site_mask_clean"][:, None]
            ].view(metadata["num_structid"], -1, 3)
            coords_ref_bs_prot = batch["features"]["res_atom_positions"][
                res_atom_mask & batch["features"]["binding_site_mask_clean"][:, None]
            ].view(metadata["num_structid"], -1, 3)
            coords_pred_lig = scores["final_coords_lig_atom"].view(
                metadata["num_structid"], -1, 3
            )
            coords_ref_lig = batch["features"]["sdf_coordinates"].view(
                metadata["num_structid"], -1, 3
            )
            coords_pred = torch.cat([coords_pred_prot, coords_pred_lig], dim=1)
            coords_ref = torch.cat([coords_ref_prot, coords_ref_lig], dim=1)
            coords_pred_bs = torch.cat([coords_pred_bs_prot, coords_pred_lig], dim=1)
            coords_ref_bs = torch.cat([coords_ref_bs_prot, coords_ref_lig], dim=1)
            n_I_per_sample = max(metadata["num_I_per_sample"])
            lig_frame_atm_idx = torch.stack(
                [
                    indexer["gather_idx_ijk_i"][indexer["gather_idx_I_ijk"]][
                        :n_I_per_sample
                    ],
                    indexer["gather_idx_ijk_j"][indexer["gather_idx_I_ijk"]][
                        :n_I_per_sample
                    ],
                    indexer["gather_idx_ijk_k"][indexer["gather_idx_I_ijk"]][
                        :n_I_per_sample
                    ],
                ],
                dim=0,
            )
            (
                global_fape_pview,
                global_fape_lview,
                normalized_fape,
            ) = self.compute_fape_from_atom37(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
                pred_lig_coords=scores["final_coords_lig_atom"],
                target_lig_coords=batch["features"]["sdf_coordinates"],
                lig_frame_atm_idx=lig_frame_atm_idx,
                split_pl_views=True,
            )
            aa_distgeom_error = self.compute_aa_distance_geometry_loss(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
            )
            lig_distgeom_error = self.compute_sm_distance_geometry_loss(
                batch,
                scores["final_coords_lig_atom"],
                batch["features"]["sdf_coordinates"],
            )
            glob_drmsd, _ = self.compute_drmsd_and_clashloss(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
                pred_lig_coords=scores["final_coords_lig_atom"],
                target_lig_coords=batch["features"]["sdf_coordinates"],
                ligatm_types=batch["features"]["atomic_numbers"].long(),
            )
            bs_drmsd, clash_error = self.compute_drmsd_and_clashloss(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
                pred_lig_coords=scores["final_coords_lig_atom"],
                target_lig_coords=batch["features"]["sdf_coordinates"],
                ligatm_types=batch["features"]["atomic_numbers"].long(),
                binding_site=True,
            )
            pli_drmsd, _ = self.compute_drmsd_and_clashloss(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
                pred_lig_coords=scores["final_coords_lig_atom"],
                target_lig_coords=batch["features"]["sdf_coordinates"],
                ligatm_types=batch["features"]["atomic_numbers"].long(),
                pl_interface=True,
            )
            distgeom_loss = (
                aa_distgeom_error.mul(lambda_weighting)
                * max(metadata["num_a_per_sample"])
                + lig_distgeom_error.mul(lambda_weighting)
                * max(metadata["num_i_per_sample"])
            ).mean() / max(metadata["num_a_per_sample"])

            fape_loss = (
                (
                    global_fape_pview
                    + global_fape_lview
                    * (
                        self.global_config.ligand_score_loss_weight
                        / self.global_config.global_score_loss_weight
                    )
                    + normalized_fape
                )
                .mul(lambda_weighting)
                .mean()
            )

            loss = (
                loss
                + fape_loss
                * self.global_config.global_score_loss_weight
                * is_native_sample
            )
            loss = (
                loss
                + glob_drmsd.mul(lambda_weighting).mean()
                * self.global_config.drmsd_loss_weight
            )
            if use_template:
                twe_drmsd = self.compute_template_weighted_centroid_drmsd(
                    batch, scores["final_coords_prot_atom_padded"]
                )
                loss = (
                    loss
                    + twe_drmsd.mul(lambda_weighting).mean()
                    * self.global_config.drmsd_loss_weight
                )
                self.log(
                    f"{stage}/drmsd_loss_weighted",
                    twe_drmsd.mul(lambda_weighting).mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
                self.log(
                    f"{stage}/drmsd_weighted",
                    twe_drmsd.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
            loss = (
                loss
                + bs_drmsd.mul(lambda_weighting).mean()
                * self.global_config.drmsd_loss_weight
            )
            loss = (
                loss
                + pli_drmsd.mul(lambda_weighting).mean()
                * self.global_config.drmsd_loss_weight
            )

            loss = (
                loss
                + distgeom_loss
                * self.global_config.local_distgeom_loss_weight
                * violation_loss_ratio
            )
            loss = (
                loss
                + clash_error.mul(lambda_weighting).mean()
                * self.global_config.clash_loss_weight
                * violation_loss_ratio
            )
            loss = (0.1 + 0.9 * prior_training) * cont_loss + (
                1 - prior_training * 0.99
            ) * loss
            loss = (
                loss + protein_distogram_loss * self.global_config.distogram_loss_weight
            )
            with torch.no_grad():
                tm_lbound = self.compute_TMscore_lbound(
                    batch,
                    scores["final_coords_prot_atom_padded"],
                    batch["features"]["res_atom_positions"],
                )
                lig_rmsd = segment_mean(
                    (
                        (coords_pred_lig - coords_pred_prot.mean(dim=1, keepdim=True))
                        - (coords_ref_lig - coords_ref_prot.mean(dim=1, keepdim=True))
                    )
                    .square()
                    .sum(dim=-1)
                    .flatten(0, 1),
                    indexer["gather_idx_i_molid"],
                    metadata["num_molid"],
                ).sqrt()
                self.log(
                    f"{stage}/tm_lbound",
                    tm_lbound.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
                self.log(
                    f"{stage}/ligand_rmsd_ubound",
                    lig_rmsd.mean().detach(),
                    on_epoch=True,
                    batch_size=lig_rmsd.shape[0],
                )
                # L1 score matching loss
                dsm_loss_global = (
                    (
                        (coords_pred - coords_pred_prot.mean(dim=1, keepdim=True))
                        - (coords_ref - coords_ref_prot.mean(dim=1, keepdim=True))
                    )
                    .square()
                    .sum(dim=-1)
                    .add(1e-2)
                    .sqrt()
                    .sub(1e-1)
                    .mean(dim=1)
                    .mul(lambda_weighting)
                )
                dsm_loss_site = (
                    (
                        (coords_pred_bs - coords_pred_bs_prot.mean(dim=1, keepdim=True))
                        - (coords_ref_bs - coords_ref_bs_prot.mean(dim=1, keepdim=True))
                    )
                    .square()
                    .sum(dim=-1)
                    .add(1e-2)
                    .sqrt()
                    .sub(1e-1)
                    .mean(dim=1)
                    .mul(lambda_weighting)
                )
                dsm_loss_ligand = (
                    (
                        (coords_pred_lig - coords_pred.mean(dim=1, keepdim=True))
                        - (coords_ref_lig - coords_ref.mean(dim=1, keepdim=True))
                    )
                    .square()
                    .sum(dim=-1)
                    .add(1e-2)
                    .sqrt()
                    .sub(1e-1)
                    .mean(dim=1)
                    .mul(lambda_weighting)
                )
                self.log(
                    f"{stage}/denoising_loss_global",
                    dsm_loss_global.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
                self.log(
                    f"{stage}/denoising_loss_site",
                    dsm_loss_site.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
                self.log(
                    f"{stage}/denoising_loss_ligand",
                    dsm_loss_ligand.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
            self.log(
                f"{stage}/drmsd_loss_global",
                glob_drmsd.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/drmsd_loss_site",
                bs_drmsd.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/drmsd_loss_pli",
                pli_drmsd.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/drmsd_global",
                glob_drmsd.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/drmsd_site",
                bs_drmsd.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/drmsd_pli",
                pli_drmsd.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_global_proteinview",
                global_fape_pview.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_global_ligandview",
                global_fape_lview.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_normalized",
                normalized_fape.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_loss",
                fape_loss.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/aa_distgeom_error",
                aa_distgeom_error.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/lig_distgeom_error",
                lig_distgeom_error.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/clash_error",
                clash_error.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/clash_loss",
                clash_error.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/distgeom_loss",
                distgeom_loss.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        else:
            coords_pred = scores["final_coords_prot_atom_padded"][res_atom_mask].view(
                metadata["num_structid"], -1, 3
            )
            coords_ref = batch["features"]["res_atom_positions"][res_atom_mask].view(
                metadata["num_structid"], -1, 3
            )
            global_fape_pview, normalized_fape = self.compute_fape_from_atom37(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
            )
            aa_distgeom_error = self.compute_aa_distance_geometry_loss(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
            )
            glob_drmsd, clash_error = self.compute_drmsd_and_clashloss(
                batch,
                scores["final_coords_prot_atom_padded"],
                batch["features"]["res_atom_positions"],
            )
            distgeom_loss = aa_distgeom_error.mul(lambda_weighting).mean()
            fape_loss = (
                (global_fape_pview + normalized_fape).mul(lambda_weighting).mean()
            )

            global_fape_pview.detach()
            loss = (
                loss
                + distgeom_loss
                * self.global_config.local_distgeom_loss_weight
                * violation_loss_ratio
            )
            loss = loss + fape_loss * self.global_config.global_score_loss_weight
            loss = (
                loss
                + glob_drmsd.mul(lambda_weighting).mean()
                * self.global_config.drmsd_loss_weight
            )
            if use_template:
                twe_drmsd = self.compute_template_weighted_centroid_drmsd(
                    batch, scores["final_coords_prot_atom_padded"]
                )
                loss = (
                    loss
                    + twe_drmsd.mul(lambda_weighting).mean()
                    * self.global_config.drmsd_loss_weight
                )
                self.log(
                    f"{stage}/drmsd_loss_weighted",
                    twe_drmsd.mul(lambda_weighting).mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
                self.log(
                    f"{stage}/drmsd_weighted",
                    twe_drmsd.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
            loss = (
                loss
                + clash_error.mul(lambda_weighting).mean()
                * self.global_config.clash_loss_weight
                * violation_loss_ratio
            )
            loss = (
                loss + protein_distogram_loss * self.global_config.distogram_loss_weight
            )

            with torch.no_grad():
                dsm_loss_global = (
                    (
                        (coords_pred - coords_pred.mean(dim=1, keepdim=True))
                        - (coords_ref - coords_ref.mean(dim=1, keepdim=True))
                    )
                    .square()
                    .sum(dim=-1)
                    .add(1e-2)
                    .sqrt()
                    .sub(1e-1)
                    .mean(dim=1)
                    .mul(lambda_weighting)
                )
                self.log(
                    f"{stage}/denoising_loss_global",
                    dsm_loss_global.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
                tm_lbound = self.compute_TMscore_lbound(
                    batch,
                    scores["final_coords_prot_atom_padded"],
                    batch["features"]["res_atom_positions"],
                )
                self.log(
                    f"{stage}/tm_lbound",
                    tm_lbound.mean().detach(),
                    on_epoch=True,
                    batch_size=batch_size,
                )
            self.log(
                f"{stage}/drmsd_loss_global",
                glob_drmsd.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/drmsd_global",
                glob_drmsd.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_global_proteinview",
                global_fape_pview.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_normalized",
                normalized_fape.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}/fape_loss",
                fape_loss.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/aa_distgeom_error",
                aa_distgeom_error.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/clash_error",
                clash_error.mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/clash_loss",
                clash_error.mul(lambda_weighting).mean().detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_violation/distgeom_loss",
                distgeom_loss.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        if not torch.isnan(loss):
            self.log(
                f"{stage}/loss",
                loss.detach(),
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=(stage != "train"),
            )
        return {
            "loss": loss,
        }

    def _assign_timestep_encodings(self, batch, t_normalized):
        # t_normalized \in [0, 1]
        indexer = batch["indexer"]
        if not isinstance(t_normalized, torch.Tensor):
            t_normalized = torch.full(
                (batch["metadata"]["num_structid"], 1),
                t_normalized,
                device=self.device,
            )
        t_prot = t_normalized[indexer["gather_idx_a_structid"]]
        batch["features"]["timestep_encoding_prot"] = t_prot

        if not batch["misc"]["protein_only"]:
            batch["features"]["timestep_encoding_lig"] = t_normalized[
                indexer["gather_idx_i_structid"]
            ]

    def _eval_confidence_estimation_losses(self, batch, batch_idx, device, stage):
        use_template = bool(random.randint(0, 1))
        if use_template:
            # Enable higher ligand diversity when using backbone template
            start_time = 1.0
        else:
            start_time = random.randint(1, 5) / 5
        with torch.no_grad():
            output_struct = self.sample_pl_complex_structures(
                batch,
                sampler="DDIM",
                num_steps=int(5 / start_time),
                start_time=start_time,
                exact_prior=True,
                use_template=use_template,
                cutoff=20.0,  # Hot logits
            )
        batch_size = batch["metadata"]["num_structid"]
        batch = self.run_confidence_estimation(batch, output_struct)
        with torch.no_grad():
            # Receptor centroids
            ref_coords = (
                (
                    batch["features"]["res_atom_positions"]
                    .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
                    .sum(dim=1)
                    .div(
                        batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None]
                        + 1e-9
                    )
                )
                .contiguous()
                .view(batch_size, -1, 3)
            )
            pred_coords = (
                (
                    output_struct["receptor_padded"]
                    .mul(batch["features"]["res_atom_mask"].bool()[:, :, None])
                    .sum(dim=1)
                    .div(
                        batch["features"]["res_atom_mask"].bool().sum(dim=1)[:, None]
                        + 1e-9
                    )
                )
                .contiguous()
                .view(batch_size, -1, 3)
            )
            # The number of effective protein atoms used in plddt calculation
            n_protatm_per_sample = pred_coords.shape[1]
            if output_struct["ligands"] is not None:
                ref_lig_coords = (
                    batch["features"]["sdf_coordinates"]
                    .contiguous()
                    .view(batch_size, -1, 3)
                )
                ref_coords = torch.cat([ref_coords, ref_lig_coords], dim=1)
                pred_lig_coords = (
                    output_struct["ligands"].contiguous().view(batch_size, -1, 3)
                )
                pred_coords = torch.cat([pred_coords, pred_lig_coords], dim=1)
            per_atom_lddt, per_atom_lddt_gram = self.compute_per_atom_lddt(
                batch, pred_coords, ref_coords
            )

        plddt_dev = (per_atom_lddt - batch["outputs"]["plddt"]).abs().mean()
        confidence_loss = (
            F.cross_entropy(
                batch["outputs"]["plddt_logits"].flatten(0, 1),
                per_atom_lddt_gram.flatten(0, 1),
                reduction="none",
            )
            .contiguous()
            .view(batch_size, -1)
        )
        conf_loss = confidence_loss.mean()
        if output_struct["ligands"] is not None:
            plddt_dev_lig = (
                (
                    per_atom_lddt.view(batch_size, -1)[:, n_protatm_per_sample:]
                    - batch["outputs"]["plddt"].view(batch_size, -1)[
                        :, n_protatm_per_sample:
                    ]
                )
                .abs()
                .mean()
            )
            conf_loss_lig = confidence_loss[:, n_protatm_per_sample:].mean()
            conf_loss = conf_loss + conf_loss_lig  # + plddt_dev_lig * 0.1
            self.log(
                f"{stage}_confidence/plddt_dev_lig",
                plddt_dev_lig.detach(),
                on_epoch=True,
                batch_size=batch_size,
            )
        self.log(
            f"{stage}_confidence/plddt_dev",
            plddt_dev.detach(),
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_confidence/loss",
            conf_loss.detach(),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=(stage != "train"),
        )
        return {
            "loss": conf_loss * self.global_config.plddt_loss_weight,
        }

    def _get_noise_schedule_vp(
        self,
        t,
        T,
        name="exp1",
        decay_rate=10.0,
        sigma_min=0.1,
        sigma_int=2.0,
        sigma_max=10.0,
    ):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device)
        if name == "cosine":
            # Cosine schedule
            alpha_t = torch.cos((t / T + 0.05) / 1.05 * math.pi / 2) ** 2
            alpha_tm = torch.cos(((t - 1) / T + 0.05) / 1.05 * math.pi / 2) ** 2
        elif name == "linear":
            # Linear schedule
            alpha_t = torch.exp(-2 * decay_rate * (t / T) ** 2)
            alpha_tm = torch.exp(-2 * decay_rate * ((t - 1) / T) ** 2)
            alpha_ratio = torch.exp(
                -2 * decay_rate * ((t / T) ** 2 - ((t - 1) / T) ** 2)
            )
        elif name == "exp":
            # Exponential schedule
            t_eff = (
                torch.exp(2 * math.log(sigma_max / sigma_min) * (t / T))
                * (sigma_min / sigma_max) ** 2
            )
            tm_eff = (
                torch.exp(2 * math.log(sigma_max / sigma_min) * ((t - 1) / T))
                * (sigma_min / sigma_max) ** 2
            )
            alpha_t = torch.exp(-((sigma_max / sigma_int) ** 2) * t_eff)
            alpha_tm = torch.exp(-((sigma_max / sigma_int) ** 2) * tm_eff)
            alpha_ratio = torch.exp(-((sigma_max / sigma_int) ** 2) * (t_eff - tm_eff))
        elif name == "exp1":
            # Normalized exponential schedule
            t_eff = 1e-3 * torch.exp(math.log(1000) * (t / T))
            tm_eff = 1e-3 * torch.exp(math.log(1000) * ((t - 1) / T))
            alpha_t = torch.exp(-2 * decay_rate * t_eff)
            alpha_tm = torch.exp(-2 * decay_rate * tm_eff)
            alpha_ratio = torch.exp(-2 * decay_rate * (t_eff - tm_eff))
        else:
            raise NotImplementedError
        beta_t = 1 - alpha_ratio
        return alpha_t, alpha_tm, beta_t

    @staticmethod
    def _get_noise_schedule_cr(t, T, sigma_max: float = 10, sigma_min: float = 0.1):
        # Centroid dynamics
        t_eff_sqrt = torch.exp(math.log(sigma_max / sigma_min) * (t / T)) * (
            sigma_min / sigma_max
        )
        tm_eff_sqrt = torch.exp(math.log(sigma_max / sigma_min) * ((t - 1) / T)) * (
            sigma_min / sigma_max
        )
        sigma_t = t_eff_sqrt * sigma_max
        sigma_tm = tm_eff_sqrt * sigma_max
        beta_t = sigma_t.square() - sigma_tm.square()
        return sigma_t, sigma_tm, beta_t

    @staticmethod
    def _get_noise_schedule_ve(t, T, max_sigma=100, min_sigma=0.1):
        pf_t = math.log(max_sigma / min_sigma)
        sigma_t = torch.exp(pf_t * t / T) * min_sigma
        sigma_tm = torch.exp(pf_t * (t - 1) / T) * min_sigma
        beta_t = sigma_t.square() - sigma_tm.square()  # * (sigma_tm / sigma_t).square()
        return sigma_t, sigma_tm, beta_t

    @staticmethod
    def _get_noise_schedule_hom(t, T, max_sigma=10):
        sigma_t = (t / T) ** (1 / 2) * max_sigma
        sigma_tm = ((t - 1) / T) ** (1 / 2) * max_sigma
        beta_t = sigma_t.square() - sigma_tm.square()
        return sigma_t, sigma_tm, beta_t

    @staticmethod
    def _forward_diffuse_vp(x0, prior_mean, prior_z, alpha_t):
        xt = (
            prior_mean
            + (x0 - prior_mean) * alpha_t.sqrt()
            + prior_z * (1 - alpha_t).sqrt()
        )
        z_eff = xt - x0  # / (1 - alpha_t).sqrt()
        return xt, z_eff

    @staticmethod
    def _forward_diffuse_ve(x0, prior_z, sigma_t):
        xt = x0 + prior_z * sigma_t
        z_eff = xt - x0
        return xt, z_eff

    @staticmethod
    def _forward_diffuse_hom(x0, prior_z, sigma_t):
        xt = x0 + prior_z * sigma_t
        z_eff = xt - x0
        return xt, z_eff

    @staticmethod
    def _reverse_diffuse_brownian_step_xpred(
        xt, x0_pred, sigma_t, sigma_tm, beta_t, DDIM=True
    ):
        zt = (xt - x0_pred) / sigma_t
        if DDIM:
            return xt - (sigma_t - sigma_tm) * zt
        else:
            return xt - 1.25 * zt * beta_t / sigma_t

    @staticmethod
    def _reverse_diffuse_hom_step_xpred(xt, x0_pred, sigma_t, beta_t):
        zt = (xt - x0_pred) / sigma_t
        return xt - zt * beta_t.sqrt()

    @staticmethod
    def _reverse_diffuse_mean_step_zpred(xt, prior_mean, z_pred, alpha_t, beta_t):
        return (xt - z_pred * beta_t / ((1 - alpha_t).sqrt()) - prior_mean) / (
            (1 - beta_t).sqrt()
        ) + prior_mean

    @staticmethod
    def _reverse_diffuse_mean_step_xpred(
        xt, prior_mean, x0_pred, alpha_t, alpha_tm, beta_t, DDIM=True
    ):
        # Nichol & Dhariwal, Eq. 11
        x0_int = x0_pred - prior_mean
        xt_int = xt - prior_mean
        if DDIM:
            return (
                (alpha_tm.sqrt() * x0_int)
                + (
                    (1 - alpha_tm).sqrt()
                    / (1 - alpha_t).sqrt()
                    * (xt_int - alpha_t.sqrt() * x0_int)
                )
                + prior_mean
            )
        else:
            return (
                (alpha_tm.sqrt() * beta_t / (1 - alpha_t) * x0_int)
                + ((1 - beta_t).sqrt() * (1 - alpha_tm) / (1 - alpha_t) * xt_int)
                + prior_mean
            )

    @staticmethod
    def _forward_diffuse_vp_marginal(x0, z_t, alpha_t):
        xt = x0 * alpha_t.sqrt() + z_t * (1 - alpha_t).sqrt()
        return xt

    @staticmethod
    def _reverse_diffuse_vp_step_ddim(xt, x0_hat, alpha_t, alpha_tm, beta_t, **kwargs):
        return (alpha_tm.sqrt() * x0_hat) + (
            (1 - alpha_tm).sqrt()
            / (1 - alpha_t).sqrt()
            * (xt - alpha_t.sqrt() * x0_hat)
        )

    @staticmethod
    def _reverse_diffuse_vp_step_sde(xt, x0_hat, alpha_t, alpha_tm, beta_t, **kwargs):
        return (
            (alpha_tm.sqrt() * beta_t / (1 - alpha_t) * x0_hat)
            + ((1 - beta_t).sqrt() * (1 - alpha_tm) / (1 - alpha_t) * xt)
            + torch.randn_like(xt) * (beta_t ** (1 / 2))
        )

    @staticmethod
    def _reverse_diffuse_vp_step_annealed_langevin_simple(
        xt, x0_hat, alpha_t, alpha_tm, beta_t, t=1.0, tm=1.0, max_inverse_temp=1.5
    ):
        denoising_step = (
            (alpha_tm.sqrt() * beta_t / (1 - alpha_t) * x0_hat)
            + ((1 - beta_t).sqrt() * (1 - alpha_tm) / (1 - alpha_t) * xt)
        ) - (1 / (1 - beta_t).sqrt()) * xt
        # TODO: Harmonic approximation
        inverse_temp_t = 1 + (1 - t) * max_inverse_temp
        inverse_temp_tm = 1 + (1 - tm) * max_inverse_temp
        out = (
            (1 / (1 - beta_t).sqrt()) * xt
            + ((1 + inverse_temp_t) / 2) * denoising_step
            + torch.randn_like(xt) * ((beta_t / inverse_temp_tm) ** (1 / 2))
        )
        return out

    @staticmethod
    def _reverse_diffuse_vp_step_annealed_langevin_semianalytic(
        xt, x0_hat, alpha_t, alpha_tm, beta_t, t=1.0, tm=1.0, max_inverse_temp=4.0
    ):
        # Harmonic approximation
        1 + (1 - t) * max_inverse_temp
        inverse_temp_tm = 1 + (1 - tm) * max_inverse_temp
        out = (
            alpha_tm.sqrt()
            * (
                1
                - ((alpha_t / alpha_tm - alpha_t) / (1 - alpha_t))
                ** ((1 + inverse_temp_tm) / 2)
            )
            * x0_hat
            + (
                ((alpha_t / alpha_tm - alpha_t) / (1 - alpha_t))
                ** ((1 + inverse_temp_tm) / 2)
                * (alpha_tm / alpha_t).sqrt()
                * xt
            )
            + (
                (torch.log(alpha_tm / alpha_t) + (alpha_t - alpha_tm))
                / (inverse_temp_tm * (1 - alpha_tm))
            ).sqrt()
            * torch.randn_like(xt)
        )
        return out

    @staticmethod
    def _reverse_diffuse_ddim_step_xpred(xt, prior_mean, x0_pred, alpha_t, alpha_tm):
        # Song 2021, Eq. 7
        x0_int = x0_pred - prior_mean
        xt_int = xt - prior_mean
        return (
            alpha_tm.sqrt() * x0_int
            + (1 - alpha_tm).sqrt()
            / (1 - alpha_t).sqrt()
            * (xt_int - alpha_t.sqrt() * x0_int)
            + prior_mean
        )

    def _forward_diffuse_plcomplex_latinp(
        self, batch, t, T, latent_converter, use_cached_noise=False, erase_data=False
    ):
        # Dimension-less internal coordinates and decay rates
        # [B, N, 3]
        x_int, lambda_int = latent_converter.to_latent(batch)
        alpha_t, alpha_tm, beta_t_lig = self._get_noise_schedule_vp(
            t,
            T,
            name="exp1",
            decay_rate=lambda_int,
        )
        z_int = torch.randn_like(x_int)
        if erase_data:
            x_int = torch.zeros_like(x_int)
        # For autoregressive prior sampling
        if use_cached_noise and (latent_converter.cached_noise is not None):
            z_int[
                :, : latent_converter.cached_noise.shape[1]
            ] = latent_converter.cached_noise
        latent_converter.cached_noise = z_int
        x_int_t = self._forward_diffuse_vp_marginal(
            x_int,
            z_int,
            alpha_t,
        )
        return latent_converter.assign_to_batch(batch, x_int_t)

    def _reverse_diffuse_plcomplex_latinp(
        self,
        batch,
        t,
        T,
        latent_converter,
        score_converter,
        sampler_step_fn,
        umeyama_correction=True,
        use_template=False,
    ):
        # Dimension-less internal coordinates and decay rates
        x_int_t, lambda_int = latent_converter.to_latent(batch)
        self._assign_timestep_encodings(batch, t / T)
        alpha_t, alpha_tm, beta_t = self._get_noise_schedule_vp(
            t,
            T,
            name="exp1",
            decay_rate=lambda_int,
        )
        batch = self.forward(
            batch,
            iter_id=score_converter.iter_id,
            contact_prediction=True,
            score=True,
            observed_block_contacts=score_converter.sampled_block_contacts,
            use_template=use_template,
        )
        if umeyama_correction:
            batch_size = batch["metadata"]["num_structid"]
            _last_pred_ca_trace = (
                batch["outputs"]["denoised_prediction"][
                    "final_coords_prot_atom_padded"
                ][:, 1]
                .view(batch_size, -1, 3)
                .detach()
            )
            if score_converter._last_pred_ca_trace is not None:
                similarity_transform = corresponding_points_alignment(
                    _last_pred_ca_trace,
                    score_converter._last_pred_ca_trace,
                    estimate_scale=False,
                )
                _last_pred_ca_trace = apply_similarity_transform(
                    _last_pred_ca_trace, *similarity_transform
                )
                protatm_padding_mask = batch["features"]["res_atom_mask"]
                pred_protatm_coords = (
                    batch["outputs"]["denoised_prediction"][
                        "final_coords_prot_atom_padded"
                    ][protatm_padding_mask]
                    .contiguous()
                    .view(batch_size, -1, 3)
                )
                aligned_pred_protatm_coords = (
                    apply_similarity_transform(
                        pred_protatm_coords, *similarity_transform
                    )
                    .contiguous()
                    .flatten(0, 1)
                )
                batch["outputs"]["denoised_prediction"][
                    "final_coords_prot_atom_padded"
                ][protatm_padding_mask] = aligned_pred_protatm_coords
                batch["outputs"]["denoised_prediction"][
                    "final_coords_prot_atom"
                ] = aligned_pred_protatm_coords
                if not batch["misc"]["protein_only"]:
                    pred_ligatm_coords = batch["outputs"]["denoised_prediction"][
                        "final_coords_lig_atom"
                    ].view(batch_size, -1, 3)
                    aligned_pred_ligatm_coords = (
                        apply_similarity_transform(
                            pred_ligatm_coords, *similarity_transform
                        )
                        .contiguous()
                        .flatten(0, 1)
                    )
                    batch["outputs"]["denoised_prediction"][
                        "final_coords_lig_atom"
                    ] = aligned_pred_ligatm_coords
            score_converter._last_pred_ca_trace = _last_pred_ca_trace

        x_int_hat_t, _ = score_converter.to_latent(batch)
        x_int_tm = sampler_step_fn(
            x_int_t, x_int_hat_t, alpha_t, alpha_tm, beta_t, t=t / T, tm=(t - 1) / T
        )
        return latent_converter.assign_to_batch(batch, x_int_tm)

    def sample_pl_complex_structures(
        self,
        batch,
        num_steps=100,
        return_summary_stats=False,
        return_all_states=False,
        sampler="DDIM",
        umeyama_correction=True,
        start_time=1.0,
        exact_prior=False,
        align_to_ground_truth=True,
        use_template=None,
        **kwargs,
    ):
        """
        Sampling protein-ligand structures.
        :param batch:
        :param num_steps:
        :param return_summary_stats:
        :param return_all_states:
        :param sampler:
        :param umeyama_correction: Apply optimal alignment between the denoised structure and previous step outputs.
        :param start_time:
        :return:
        """
        T = num_steps
        start_step = int((1 - start_time) * T)  # 0->T
        if use_template is None:
            use_template = self.global_config.use_template

        indexer = batch["indexer"]
        metadata = batch["metadata"]
        res_atom_mask = batch["features"]["res_atom_mask"].bool()

        if (
            "num_molid" in batch["metadata"].keys()
            and batch["metadata"]["num_molid"] > 0
        ):
            batch["misc"]["protein_only"] = False
        else:
            batch["misc"]["protein_only"] = True

        forward_lat_converter = self._resolve_latent_converter(
            [
                ("features", "res_atom_positions"),
                ("features", "input_protein_coords"),
            ],
            [("features", "sdf_coordinates"), ("features", "input_ligand_coords")],
        )
        reverse_lat_converter = self._resolve_latent_converter(
            [
                ("features", "input_protein_coords"),
                ("features", "input_protein_coords"),
            ],
            [
                ("features", "input_ligand_coords"),
                ("features", "input_ligand_coords"),
            ],
        )
        reverse_score_converter = self._resolve_latent_converter(
            [
                (
                    "outputs",
                    "denoised_prediction",
                    "final_coords_prot_atom_padded",
                ),
                None,
            ],
            [
                (
                    "outputs",
                    "denoised_prediction",
                    "final_coords_lig_atom",
                ),
                None,
            ],
        )

        with torch.no_grad():
            if not batch["misc"]["protein_only"]:
                # Autoregressive block contact map prior
                if exact_prior:
                    batch = self._prepare_protein_patch_indexers(batch)
                    ref_dists_au, contact_logit_matrix = self._eval_true_contact_maps(
                        batch, **kwargs
                    )
                else:
                    batch["misc"]["protein_only"] = True
                    batch = self._forward_diffuse_plcomplex_latinp(
                        batch,
                        T - start_step,
                        T,
                        forward_lat_converter,
                        erase_data=(start_time >= 1.0),
                    )
                    batch["misc"]["protein_only"] = False
                    self._assign_timestep_encodings(batch, (T - start_step) / T)
                    # Sample the categorical contact encodings under the hood
                    batch = self.forward(
                        batch,
                        contact_prediction=True,
                        infer_geometry_prior=True,
                        use_template=use_template,
                    )
                    # Sample initial ligand coordinates from the geometry prior
                    contact_logit_matrix = batch["outputs"]["geometry_prior_L"]

                sampled_lig_res_anchor_mask = self._sample_res_rowmask_from_contacts(
                    batch, contact_logit_matrix
                )
                num_cont_to_sample = max(metadata["num_I_per_sample"])
                sampled_block_contacts = None
                for _ in range(num_cont_to_sample):
                    sampled_block_contacts = self._sample_reslig_contact_matrix(
                        batch, contact_logit_matrix, last=sampled_block_contacts
                    )
                forward_lat_converter.lig_res_anchor_mask = sampled_lig_res_anchor_mask
                reverse_lat_converter.lig_res_anchor_mask = sampled_lig_res_anchor_mask
                reverse_score_converter.lig_res_anchor_mask = (
                    sampled_lig_res_anchor_mask
                )
                reverse_score_converter.iter_id = num_cont_to_sample
                reverse_score_converter.sampled_block_contacts = sampled_block_contacts
            else:
                reverse_score_converter.iter_id = 0
                reverse_score_converter.sampled_block_contacts = None

        if sampler == "DDIM":
            sampling_step_fn = self._reverse_diffuse_vp_step_ddim
        elif sampler == "VPSDE":
            sampling_step_fn = self._reverse_diffuse_vp_step_sde
        elif sampler == "simulated_annealing_simple":
            sampling_step_fn = self._reverse_diffuse_vp_step_annealed_langevin_simple
        elif sampler == "langevin_simulated_annealing":
            sampling_step_fn = (
                self._reverse_diffuse_vp_step_annealed_langevin_semianalytic
            )
        else:
            raise NotImplementedError

        with torch.no_grad():
            batch = self._forward_diffuse_plcomplex_latinp(
                batch,
                T - start_step,
                T,
                forward_lat_converter,
                # use_cached_noise=True,
                erase_data=(start_time >= 1.0),
            )

            if return_all_states:
                all_frames = []
            # We follow https://arxiv.org/pdf/2006.11239.pdf for all symbolic conventions
            for time_step in tqdm.tqdm(
                range(start_step, T), desc=f"Structure generation using {sampler}"
            ):
                t = T - time_step
                batch = self._reverse_diffuse_plcomplex_latinp(
                    batch,
                    t,
                    T,
                    reverse_lat_converter,
                    reverse_score_converter,
                    sampling_step_fn,
                    umeyama_correction=umeyama_correction,
                    use_template=use_template,
                )
                if return_all_states:
                    # all_frames.append(
                    #     {
                    #         "ligands": batch["features"]["input_ligand_coords"],
                    #         "receptor": batch["features"]["input_protein_coords"][
                    #             res_atom_mask
                    #         ],
                    #         "receptor_padded": batch["features"][
                    #             "input_protein_coords"
                    #         ],
                    #     }
                    # )
                    all_frames.append(
                        {
                            "ligands": batch["outputs"]["denoised_prediction"][
                                "final_coords_lig_atom"
                            ],
                            "receptor": batch["outputs"]["denoised_prediction"][
                                "final_coords_prot_atom"
                            ],
                            "receptor_padded": batch["outputs"]["denoised_prediction"][
                                "final_coords_prot_atom_padded"
                            ],
                        }
                    )

            mean_x1 = batch["outputs"]["denoised_prediction"]["final_coords_lig_atom"]
            mean_x2_padded = batch["outputs"]["denoised_prediction"][
                "final_coords_prot_atom_padded"
            ]
            protatm_padding_mask = batch["features"]["res_atom_mask"]
            mean_x2 = mean_x2_padded[protatm_padding_mask]
            if align_to_ground_truth:
                batch_size = batch["metadata"]["num_structid"]
                similarity_transform = corresponding_points_alignment(
                    mean_x2_padded[:, 1].view(batch_size, -1, 3),
                    batch["features"]["res_atom_positions"][:, 1].view(
                        batch_size, -1, 3
                    ),
                    estimate_scale=False,
                )
                mean_x2 = (
                    apply_similarity_transform(
                        mean_x2.view(batch_size, -1, 3), *similarity_transform
                    )
                    .contiguous()
                    .flatten(0, 1)
                )
                mean_x2_padded[protatm_padding_mask] = mean_x2
                if mean_x1 is not None:
                    mean_x1 = (
                        apply_similarity_transform(
                            mean_x1.view(batch_size, -1, 3), *similarity_transform
                        )
                        .contiguous()
                        .flatten(0, 1)
                    )

            if return_all_states:
                all_frames.append(
                    {
                        "ligands": mean_x1,
                        "receptor": mean_x2,
                        "receptor_padded": mean_x2_padded,
                    }
                )
            protein_fape, normalized_fape = self.compute_fape_from_atom37(
                batch,
                mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            tm_lbound = self.compute_TMscore_lbound(
                batch,
                mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            tm_lbound_mirrored = self.compute_TMscore_lbound(
                batch,
                -mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            tm_aligned_ca = self.compute_TMscore_raw(
                batch,
                mean_x2_padded[:, 1],
                batch["features"]["res_atom_positions"][:, 1],
            )
            lddt_ca = self.compute_lddt_ca(
                batch,
                mean_x2_padded,
                batch["features"]["res_atom_positions"],
            )
            ret = {
                "FAPE_protein": protein_fape,
                "TM_aligned_ca": tm_aligned_ca,
                "TM_lbound": tm_lbound,
                "TM_lbound_mirrored": tm_lbound_mirrored,
                "lDDT-Ca": lddt_ca,
            }
            if mean_x1 is not None:
                n_I_per_sample = max(metadata["num_I_per_sample"])
                lig_frame_atm_idx = torch.stack(
                    [
                        indexer["gather_idx_ijk_i"][indexer["gather_idx_I_ijk"]][
                            :n_I_per_sample
                        ],
                        indexer["gather_idx_ijk_j"][indexer["gather_idx_I_ijk"]][
                            :n_I_per_sample
                        ],
                        indexer["gather_idx_ijk_k"][indexer["gather_idx_I_ijk"]][
                            :n_I_per_sample
                        ],
                    ],
                    dim=0,
                )
                prot_fape, lig_fape, normalized_fape = self.compute_fape_from_atom37(
                    batch,
                    mean_x2_padded,
                    batch["features"]["res_atom_positions"],
                    pred_lig_coords=mean_x1,
                    target_lig_coords=batch["features"]["sdf_coordinates"],
                    lig_frame_atm_idx=lig_frame_atm_idx,
                    split_pl_views=True,
                )
                coords_pred_prot = mean_x2_padded[res_atom_mask].view(
                    metadata["num_structid"], -1, 3
                )
                coords_ref_prot = batch["features"]["res_atom_positions"][
                    res_atom_mask
                ].view(metadata["num_structid"], -1, 3)
                coords_pred_lig = mean_x1.view(metadata["num_structid"], -1, 3)
                coords_ref_lig = batch["features"]["sdf_coordinates"].view(
                    metadata["num_structid"], -1, 3
                )
                lig_rmsd = segment_mean(
                    (
                        (coords_pred_lig - coords_pred_prot.mean(dim=1, keepdim=True))
                        - (coords_ref_lig - coords_ref_prot.mean(dim=1, keepdim=True))
                    )
                    .square()
                    .sum(dim=-1)
                    .flatten(0, 1),
                    indexer["gather_idx_i_molid"],
                    metadata["num_molid"],
                ).sqrt()
                lig_centroid_distance = (
                    segment_mean(
                        (
                            coords_pred_lig - coords_pred_prot.mean(dim=1, keepdim=True)
                        ).flatten(0, 1),
                        indexer["gather_idx_i_molid"],
                        metadata["num_molid"],
                    )
                    - segment_mean(
                        (
                            coords_ref_lig - coords_ref_prot.mean(dim=1, keepdim=True)
                        ).flatten(0, 1),
                        indexer["gather_idx_i_molid"],
                        metadata["num_molid"],
                    )
                ).norm(dim=-1)
                lig_hit_score_1A = (lig_rmsd < 1.0).float()
                lig_hit_score_2A = (lig_rmsd < 2.0).float()
                lig_hit_score_4A = (lig_rmsd < 4.0).float()
                lddt_pli = self.compute_lddt_pli(
                    batch,
                    mean_x2_padded,
                    batch["features"]["res_atom_positions"],
                    mean_x1,
                    batch["features"]["sdf_coordinates"],
                )
                ret.update(
                    {
                        "ligand_RMSD": lig_rmsd,
                        "ligand_centroid_distance": lig_centroid_distance,
                        "lDDT-pli": lddt_pli,
                        "FAPE_ligview": lig_fape,
                        "ligand_hit_score_1A": lig_hit_score_1A,
                        "ligand_hit_score_2A": lig_hit_score_2A,
                        "ligand_hit_score_4A": lig_hit_score_4A,
                    }
                )

        if return_summary_stats:
            return ret

        if return_all_states:
            return all_frames

        ret.update(
            {
                "ligands": mean_x1,
                "receptor": mean_x2,
                "receptor_padded": mean_x2_padded,
            }
        )

        return ret

    def epoch_end(self, outs, stage):
        if self.global_config.task_type == "LBA":
            self.log(f"{stage}_r2", self._compute_epoch_r2(outs))
            self.log(f"{stage}_pearson_r", self._compute_pearson_r(outs))

    def training_step(self, batch, batch_idx):
        self.train()
        return self.step(batch, batch_idx, "train")

    def training_epoch_end(self, outs):
        self.epoch_end(outs, "train")

    def validation_step(self, batch, batch_idx):
        self.eval()
        if (self.current_epoch % 5) == 0:
            sampling_stats = self.sample_pl_complex_structures(
                batch,
                sampler="DDIM",
                num_steps=10,
                start_time=1.0,
                return_summary_stats=True,
                exact_prior=False,
            )
            for metric_name in sampling_stats.keys():
                log_stat = sampling_stats[metric_name].mean().detach()
                batch_size = sampling_stats[metric_name].shape[0]
                self.log(
                    f"val_sampling/{metric_name}",
                    log_stat,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            sampling_stats = self.sample_pl_complex_structures(
                batch,
                sampler="DDIM",
                num_steps=10,
                start_time=1.0,
                return_summary_stats=True,
                exact_prior=False,
                use_template=False,
            )
            for metric_name in sampling_stats.keys():
                log_stat = sampling_stats[metric_name].mean().detach()
                batch_size = sampling_stats[metric_name].shape[0]
                self.log(
                    f"val_sampling_notemplate/{metric_name}",
                    log_stat,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            sampling_stats = self.sample_pl_complex_structures(
                batch,
                sampler="DDIM",
                num_steps=10,
                start_time=1.0,
                return_summary_stats=True,
                exact_prior=True,
            )
            for metric_name in sampling_stats.keys():
                log_stat = sampling_stats[metric_name].mean().detach()
                batch_size = sampling_stats[metric_name].shape[0]
                self.log(
                    f"val_sampling_trueprior/{metric_name}",
                    log_stat,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )
            return self.step(batch, batch_idx, "val")

    def validation_epoch_end(self, outs):
        self.epoch_end(outs, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        sampling_stats = self.sample_pl_complex_structures(
            batch,
            sampler="DDIM",
            num_steps=20,
            start_time=1.0,
            return_summary_stats=True,
            exact_prior=False,
        )
        for metric_name in sampling_stats.keys():
            log_stat = sampling_stats[metric_name].mean().detach()
            batch_size = sampling_stats[metric_name].shape[0]
            self.log(
                f"test_sampling/{metric_name}",
                log_stat,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        sampling_stats = self.sample_pl_complex_structures(
            batch,
            sampler="DDIM",
            num_steps=20,
            start_time=1.0,
            return_summary_stats=True,
            exact_prior=True,
        )
        for metric_name in sampling_stats.keys():
            log_stat = sampling_stats[metric_name].mean().detach()
            batch_size = sampling_stats[metric_name].shape[0]
            self.log(
                f"test_sampling_trueprior/{metric_name}",
                log_stat,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        return self.step(batch, batch_idx, "test")

    def test_epoch_end(self, outs):
        self.epoch_end(outs, "test")
        if self.global_config.task_type == "LBA":
            self._log_predictions(outs)

    def _log_predictions(self, outs):
        preds, labels = self._get_epoch_outputs(outs)
        data = list(zip(preds, labels))
        table = wandb.Table(data=data, columns=["prediction", "ground_truth"])
        self.logger.experiment.log({"test_samples_table": table})
        self.logger.experiment.log(
            {
                "test_samples_plot": wandb.plot.scatter(
                    table, "predicted", "ground_truth", title="scatter plot"
                )
            }
        )

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break
        if not valid_gradients:
            print(
                f"detected inf or nan values in gradients. not updating model parameters"
            )
            self.zero_grad()

    def on_save_checkpoint(self, checkpoint):
        # Omit PLM params to save disk space
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("plm."):
                del checkpoint["state_dict"][key]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # Learning rate warming up
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.global_config.init_learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.global_config.init_learning_rate
        )
        # 4 cycles
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            self.global_config.max_epoch // 15,
            T_mult=2,
            eta_min=1e-8,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }


if __name__ == "__main__":
    import pprint
    import sys

    from neuralplexer.data.pipeline import inplace_to_torch, process_smiles
    from neuralplexer.model.config import (
        _attach_ligand_pretraining_task_config, get_base_config)

    torch.set_printoptions(profile="full")

    pp = pprint.PrettyPrinter(indent=4)
    # Testing featurization
    test_smiles = sys.argv[1]
    test_metadata, test_indexer, test_features = process_smiles(
        test_smiles, max_pi_length=4, only_2d=False
    )
    test_features = {k: torch.Tensor(v) for k, v in test_features.items()}
    test_indexer = {k: torch.LongTensor(v) for k, v in test_indexer.items()}
    test_sample = inplace_to_torch(
        {
            "metadata": test_metadata,
            "indexer": test_indexer,
            "features": test_features,
        }
    )
    config = get_base_config()
    config = _attach_ligand_pretraining_task_config(config)
    # init model
    model = MolPretrainingWrapper(config)
    pred = model(test_sample)
    pp.pprint(pred)
