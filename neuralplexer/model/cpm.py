import random

import torch
from torch import nn

from neuralplexer.model.common import GELUMLP
from neuralplexer.model.embedding import (GaussianFourierEncoding1D,
                                          RelativeGeometryEncoding)
from neuralplexer.model.modules import (BiDirectionalTriangleAttention,
                                        TransformerLayer)
from neuralplexer.util.frame import cartesian_to_internal, get_frame_matrix
from neuralplexer.util.tensorgraph import make_multi_relation_graph_batcher


class ProtFormer(nn.Module):
    """Protein relational reasoning with downsampled edges"""

    def __init__(
        self,
        dim,
        pair_dim,
        n_blocks=4,
        n_heads=8,
        n_protein_patches=32,
        dropout=0.0,
    ):
        super(ProtFormer, self).__init__()
        self.dim = dim
        self.pair_dim = pair_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.time_encoding = GaussianFourierEncoding1D(16)
        self.res_in_mlp = GELUMLP(dim + 32, dim, dropout=dropout)
        self.chain_pos_encoding = GaussianFourierEncoding1D(self.pair_dim // 4)

        self.rel_geom_enc = RelativeGeometryEncoding(15, self.pair_dim)
        self.template_nenc = GELUMLP(64 + 37 * 3, self.dim, n_hidden_feats=128)
        self.template_eenc = RelativeGeometryEncoding(15, self.pair_dim)
        self.template_binding_site_enc = nn.Linear(1, 64, bias=False)
        self.pp_edge_embed = GELUMLP(
            pair_dim + self.pair_dim // 4 * 2 + dim * 2,
            self.pair_dim,
            n_hidden_feats=dim,
            dropout=dropout,
        )

        self.graph_stacks = nn.ModuleList(
            [
                TransformerLayer(
                    dim,
                    n_heads,
                    head_dim=pair_dim // n_heads,
                    edge_channels=pair_dim,
                    edge_update=True,
                    dropout=dropout,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.ABab_mha = TransformerLayer(
            pair_dim,
            n_heads,
            bidirectional=True,
        )

        self.triangle_stacks = nn.ModuleList(
            [
                BiDirectionalTriangleAttention(pair_dim, pair_dim // n_heads, n_heads)
                for _ in range(self.n_blocks)
            ]
        )
        self.graph_relations = [
            (
                "residue_to_residue",
                "gather_idx_ab_a",
                "gather_idx_ab_b",
                "prot_res",
                "prot_res",
            ),
            (
                "sampled_residue_to_sampled_residue",
                "gather_idx_AB_a",
                "gather_idx_AB_b",
                "prot_res",
                "prot_res",
            ),
        ]

    def _compute_chain_pe(
        self,
        residue_index,
        res_chain_index,
        src_rid,
        dst_rid,
    ):
        chain_disp = residue_index[src_rid] - residue_index[dst_rid]
        chain_rope = self.chain_pos_encoding(chain_disp.div(8).unsqueeze(-1)).div(
            chain_disp.div(8).abs().add(1).unsqueeze(-1)
        )
        # Mask cross-chain entries
        chain_mask = res_chain_index[src_rid] == res_chain_index[dst_rid]
        chain_rope = chain_rope * chain_mask[..., None]
        return chain_rope

    def _compute_chain_pair_pe(
        self,
        residue_index,
        res_chain_index,
        AB_broadcasted_rid,
        ab_rid,
        AB_broadcasted_cid,
        ab_cid,
    ):
        chain_disp_row = residue_index[AB_broadcasted_rid] - residue_index[ab_rid]
        chain_disp_col = residue_index[AB_broadcasted_cid] - residue_index[ab_cid]
        chain_disp = torch.stack([chain_disp_row, chain_disp_col], dim=-1)
        chain_rope = self.chain_pos_encoding(chain_disp.div(8).unsqueeze(-1)).div(
            chain_disp.div(8).abs().add(1).unsqueeze(-1)
        )
        # Mask cross-chain entries
        chain_mask_row = res_chain_index[AB_broadcasted_rid] == res_chain_index[ab_rid]
        chain_mask_col = res_chain_index[AB_broadcasted_cid] == res_chain_index[ab_cid]
        chain_mask = torch.stack([chain_mask_row, chain_mask_col], dim=-1)
        chain_rope = (chain_rope * chain_mask[..., None]).flatten(-2, -1)
        return chain_rope

    def _eval_protein_template_encodings(self, batch, edge_idx, use_plddt=False):
        with torch.no_grad():
            template_bb_coords = batch["features"]["template_atom_positions"][:, :3]
            template_bb_frames = get_frame_matrix(
                template_bb_coords[:, 0, :],
                template_bb_coords[:, 1, :],
                template_bb_coords[:, 2, :],
            )
            # Add template local representations & lddt
            template_local_coords = cartesian_to_internal(
                batch["features"]["template_atom_positions"],
                template_bb_frames.unsqueeze(1),
            )
            template_local_coords[~batch["features"]["template_atom37_mask"].bool()] = 0
            # if use_plddt:
            #     template_plddt_enc = F.one_hot(
            #         torch.bucketize(
            #             batch["features"]["template_pLDDT"],
            #             torch.linspace(0, 1, 65, device=template_bb_coords.device)[:-1],
            #             right=True,
            #         )
            #         - 1,
            #         num_classes=64,
            #     )
            # else:
            #     template_plddt_enc = torch.zeros(
            #         template_local_coords.shape[0], 64, device=template_bb_coords.device
            #     )
            if self.training:
                use_sidechain_coords = random.randint(0, 1)
                template_local_coords = template_local_coords * use_sidechain_coords
                # use_plddt_input = random.randint(0, 1)
                # template_plddt_enc = template_plddt_enc * use_plddt_input
            if "binding_site_mask" in batch["features"].keys():
                # Externally-specified binding residue list
                binding_site_enc = self.template_binding_site_enc(
                    batch["features"]["binding_site_mask"][:, None].float()
                )
            else:
                binding_site_enc = torch.zeros(
                    template_local_coords.shape[0], 64, device=template_bb_coords.device
                )
            template_nfeat = self.template_nenc(
                torch.cat(
                    [template_local_coords.flatten(-2, -1), binding_site_enc], dim=-1
                )
            )
            template_efeat = self.template_eenc(template_bb_frames, edge_idx)
            template_alignment_mask = batch["features"][
                "template_alignment_mask"
            ].float()
        if self.training:
            # template_alignment_mask = template_alignment_mask * use_template
            nomasking_rate = random.randint(9, 10) / 10
            template_alignment_mask = template_alignment_mask * (
                torch.rand_like(template_alignment_mask) < nomasking_rate
            )
        template_nfeat = template_nfeat * template_alignment_mask.unsqueeze(-1)
        template_efeat = (
            template_efeat
            * template_alignment_mask[edge_idx[0]].unsqueeze(-1)
            * template_alignment_mask[edge_idx[1]].unsqueeze(-1)
        )
        return template_nfeat, template_efeat

    def forward(self, batch, **kwargs):
        return self.forward_prot_sample(batch, **kwargs)

    def forward_prot_sample(
        self,
        batch,
        embed_coords=True,
        in_attr_suffix="",
        out_attr_suffix="",
        use_template=False,
        use_plddt=False,
    ):
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        device = features["res_type"].device

        time_encoding = self.time_encoding(features["timestep_encoding_prot"])
        if not embed_coords:
            time_encoding = torch.zeros_like(time_encoding)

        residue_rep = (
            self.res_in_mlp(
                torch.cat(
                    [
                        features["res_embedding_in"],
                        time_encoding,
                    ],
                    dim=1,
                )
            )
            + features["res_embedding_in"]
        )
        batch_size = metadata["num_structid"]

        # Prepare indexers
        # Use max to ensure segmentation faults are 100% invoked
        # in case there are any bad indices
        max(metadata["num_a_per_sample"])
        n_protein_patches = batch["metadata"]["n_prot_patches_per_sample"]

        indexer["gather_idx_pid_b"] = indexer["gather_idx_pid_a"]
        # Evaluate gather_idx_AB_a and gather_idx_AB_b
        # Assign a to rows and b to columns
        # Simple broadcasting for single-structure batches
        indexer["gather_idx_AB_a"] = (
            indexer["gather_idx_pid_a"]
            .view(batch_size, n_protein_patches)[:, :, None]
            .expand(-1, -1, n_protein_patches)
            .contiguous()
            .flatten()
        )
        indexer["gather_idx_AB_b"] = (
            indexer["gather_idx_pid_b"]
            .view(batch_size, n_protein_patches)[:, None, :]
            .expand(-1, n_protein_patches, -1)
            .contiguous()
            .flatten()
        )

        # Handle all batch offsets here
        graph_batcher = make_multi_relation_graph_batcher(
            self.graph_relations, indexer, metadata
        )
        merged_edge_idx = graph_batcher.collate_idx_list(indexer)

        input_protein_coords_padded = features["input_protein_coords"]
        backbone_frames = get_frame_matrix(
            input_protein_coords_padded[:, 0, :],
            input_protein_coords_padded[:, 1, :],
            input_protein_coords_padded[:, 2, :],
        )
        batch["features"]["backbone_frames"] = backbone_frames
        # Adding geometrical info to pair representations

        chain_pe = self._compute_chain_pe(
            features["residue_index"],
            features["res_chain_id"],
            merged_edge_idx[0],
            merged_edge_idx[1],
        )
        geometry_pe = self.rel_geom_enc(backbone_frames, merged_edge_idx)
        if not embed_coords:
            geometry_pe = torch.zeros_like(geometry_pe)
        merged_edge_reps = self.pp_edge_embed(
            torch.cat(
                [
                    geometry_pe,
                    chain_pe,
                    residue_rep[merged_edge_idx[0]],
                    residue_rep[merged_edge_idx[1]],
                ],
                dim=-1,
            )
        )
        if use_template:
            (
                template_res_encodings,
                template_geom_encodings,
            ) = self._eval_protein_template_encodings(
                batch, merged_edge_idx, use_plddt=use_plddt
            )
            residue_rep = residue_rep + template_res_encodings
            merged_edge_reps = merged_edge_reps + template_geom_encodings
        edge_reps = graph_batcher.offload_edge_attr(merged_edge_reps)

        node_reps = {"prot_res": residue_rep}

        gather_idx_res_protpatch = indexer["gather_idx_a_pid"]
        # Pointer: AB->AB, ab->AB
        gather_idx_ab_AB = (
            indexer["gather_idx_ab_structid"] * n_protein_patches**2
            + (gather_idx_res_protpatch % n_protein_patches)[indexer["gather_idx_ab_a"]]
            * n_protein_patches
            + (gather_idx_res_protpatch % n_protein_patches)[indexer["gather_idx_ab_b"]]
        )

        # Intertwine graph iterations and triangle iterations
        for block_id in range(self.n_blocks):
            # Communicate between atomistic and patch resolutions
            # Up-sampling for interface edge embeddings
            rec_pair_rep = edge_reps["residue_to_residue"]
            AB_grid_attr_flat = edge_reps["sampled_residue_to_sampled_residue"]
            # Upper-left block: intra-window visual-attention
            # Cross-attention between random and  grid edges
            rec_pair_rep, AB_grid_attr_flat = self.ABab_mha(
                rec_pair_rep,
                AB_grid_attr_flat,
                (
                    torch.arange(metadata["num_ab"], device=device),
                    gather_idx_ab_AB,
                ),
            )
            AB_grid_attr = AB_grid_attr_flat.view(
                batch_size,
                n_protein_patches,
                n_protein_patches,
                self.pair_dim,
            )

            # Inter-patch triangle attentions, refining intermolecular edges
            _, AB_grid_attr = self.triangle_stacks[block_id](
                AB_grid_attr,
                AB_grid_attr,
                AB_grid_attr.unsqueeze(-4),
            )

            # Transfer grid-formatted representations to edges
            edge_reps["residue_to_residue"] = rec_pair_rep
            edge_reps["sampled_residue_to_sampled_residue"] = AB_grid_attr.flatten(0, 2)
            merged_node_reps = graph_batcher.collate_node_attr(node_reps)
            merged_edge_reps = graph_batcher.collate_edge_attr(edge_reps)

            # Graph transformer iteration
            _, merged_node_reps, merged_edge_reps = self.graph_stacks[block_id](
                merged_node_reps,
                merged_node_reps,
                merged_edge_idx,
                merged_edge_reps,
            )
            node_reps = graph_batcher.offload_node_attr(merged_node_reps)
            edge_reps = graph_batcher.offload_edge_attr(merged_edge_reps)

        batch["features"][f"rec_res_attr{out_attr_suffix}"] = node_reps["prot_res"]
        batch["features"][f"res_res_pair_attr{out_attr_suffix}"] = edge_reps[
            "residue_to_residue"
        ]
        batch["features"][f"res_res_grid_attr_flat{out_attr_suffix}"] = edge_reps[
            "sampled_residue_to_sampled_residue"
        ]
        batch["indexer"]["gather_idx_AB_a"] = indexer["gather_idx_AB_a"]
        batch["indexer"]["gather_idx_AB_b"] = indexer["gather_idx_AB_b"]
        batch["indexer"]["gather_idx_ab_AB"] = gather_idx_ab_AB
        return batch


class BindingFormer(ProtFormer):
    """Edge inference on protein-ligand graphs"""

    def __init__(
        self,
        dim,
        pair_dim,
        n_blocks=4,
        n_heads=8,
        n_protein_patches=32,
        n_ligand_patches=16,
        dropout=0.0,
    ):
        super(BindingFormer, self).__init__(
            dim,
            pair_dim,
            n_blocks,
            n_heads,
            n_protein_patches,
            dropout,
        )
        self.dim = dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.n_ligand_patches = n_ligand_patches
        self.pl_edge_embed = GELUMLP(
            dim * 2, self.pair_dim, n_hidden_feats=dim, dropout=dropout
        )
        self.AaJ_mha = TransformerLayer(pair_dim, n_heads, bidirectional=True)

        self.graph_relations = [
            (
                "residue_to_residue",
                "gather_idx_ab_a",
                "gather_idx_ab_b",
                "prot_res",
                "prot_res",
            ),
            (
                "sampled_residue_to_sampled_residue",
                "gather_idx_AB_a",
                "gather_idx_AB_b",
                "prot_res",
                "prot_res",
            ),
            (
                "sampled_residue_to_sampled_lig_triplet",
                "gather_idx_AJ_a",
                "gather_idx_AJ_J",
                "prot_res",
                "lig_trp",
            ),
            (
                "sampled_lig_triplet_to_sampled_residue",
                "gather_idx_AJ_J",
                "gather_idx_AJ_a",
                "lig_trp",
                "prot_res",
            ),
            (
                "residue_to_sampled_lig_triplet",
                "gather_idx_aJ_a",
                "gather_idx_aJ_J",
                "prot_res",
                "lig_trp",
            ),
            (
                "sampled_lig_triplet_to_residue",
                "gather_idx_aJ_J",
                "gather_idx_aJ_a",
                "lig_trp",
                "prot_res",
            ),
            (
                "sampled_lig_triplet_to_sampled_lig_triplet",
                "gather_idx_IJ_I",
                "gather_idx_IJ_J",
                "lig_trp",
                "lig_trp",
            ),
        ]

    def forward(
        self,
        batch,
        observed_block_contacts=None,
        in_attr_suffix="",
        out_attr_suffix="",
    ):
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        device = features["res_type"].device
        # Synchronize with a language model
        residue_rep = features[f"rec_res_attr{in_attr_suffix}"]
        rec_pair_rep = features[f"res_res_pair_attr{in_attr_suffix}"]
        # Inherit the last-layer pair representations from protein encoder
        AB_grid_attr_flat = features[f"res_res_grid_attr_flat{in_attr_suffix}"]

        # Prepare indexers
        batch_size = metadata["num_structid"]
        n_a_per_sample = max(metadata["num_a_per_sample"])
        n_protein_patches = batch["metadata"]["n_prot_patches_per_sample"]

        if not batch["misc"]["protein_only"]:
            n_ligand_patches = max(metadata["num_I_per_sample"])
            max(metadata["num_molid_per_sample"])
            lig_frame_rep = features[f"lig_trp_attr{in_attr_suffix}"]
            UI_grid_attr = features["lig_af_grid_attr_projected"]
            IJ_grid_attr = (UI_grid_attr + UI_grid_attr.transpose(1, 2)) / 2

            aJ_grid_attr = self.pl_edge_embed(
                torch.cat(
                    [
                        residue_rep.view(batch_size, n_a_per_sample, self.dim)[
                            :, :, None
                        ].expand(-1, -1, n_ligand_patches, -1),
                        lig_frame_rep.view(batch_size, n_ligand_patches, self.dim)[
                            :, None, :
                        ].expand(-1, n_a_per_sample, -1, -1),
                    ],
                    dim=-1,
                )
            )
            AJ_grid_attr = IJ_grid_attr.new_zeros(
                batch_size, n_protein_patches, n_ligand_patches, self.pair_dim
            )
            gather_idx_I_I = torch.arange(
                batch_size * n_ligand_patches, device=AJ_grid_attr.device
            )
            gather_idx_a_a = torch.arange(
                batch_size * n_a_per_sample, device=AJ_grid_attr.device
            )
            # Note: off-diagonal (AJ) blocks are zero-initialized in the prior stack
            indexer["gather_idx_IJ_I"] = (
                gather_idx_I_I.view(batch_size, n_ligand_patches)[:, :, None]
                .expand(-1, -1, n_ligand_patches)
                .contiguous()
                .flatten()
            )
            indexer["gather_idx_IJ_J"] = (
                gather_idx_I_I.view(batch_size, n_ligand_patches)[:, None, :]
                .expand(-1, n_ligand_patches, -1)
                .contiguous()
                .flatten()
            )
            indexer["gather_idx_AJ_a"] = (
                indexer["gather_idx_pid_a"]
                .view(batch_size, n_protein_patches)[:, :, None]
                .expand(-1, -1, n_ligand_patches)
                .contiguous()
                .flatten()
            )
            indexer["gather_idx_AJ_J"] = (
                gather_idx_I_I.view(batch_size, n_ligand_patches)[:, None, :]
                .expand(-1, n_protein_patches, -1)
                .contiguous()
                .flatten()
            )
            indexer["gather_idx_aJ_a"] = (
                gather_idx_a_a.view(batch_size, n_a_per_sample)[:, :, None]
                .expand(-1, -1, n_ligand_patches)
                .contiguous()
                .flatten()
            )
            indexer["gather_idx_aJ_J"] = (
                gather_idx_I_I.view(batch_size, n_ligand_patches)[:, None, :]
                .expand(-1, n_a_per_sample, -1)
                .contiguous()
                .flatten()
            )
            batch["indexer"] = indexer

            if observed_block_contacts is not None:
                # Generative feedback from block one-hot sampling
                # AJ_grid_attr = (
                #     AJ_grid_attr
                #     + observed_block_contacts.transpose(1, 2)
                #     .contiguous()
                #     .flatten(0, 1)[indexer["gather_idx_I_molid"]]
                #     .view(batch_size, n_ligand_patches, n_protein_patches, -1)
                #     .transpose(1, 2)
                #     .contiguous()
                # )
                AJ_grid_attr = AJ_grid_attr + observed_block_contacts

            graph_batcher = make_multi_relation_graph_batcher(
                self.graph_relations, indexer, metadata
            )
            merged_edge_idx = graph_batcher.collate_idx_list(indexer)
            node_reps = {
                "prot_res": residue_rep,
                "lig_trp": lig_frame_rep,
            }
            edge_reps = {
                "residue_to_residue": rec_pair_rep,
                "sampled_residue_to_sampled_residue": AB_grid_attr_flat,
                "sampled_lig_triplet_to_sampled_residue": AJ_grid_attr.flatten(0, 2),
                "sampled_residue_to_sampled_lig_triplet": AJ_grid_attr.flatten(0, 2),
                "sampled_lig_triplet_to_residue": aJ_grid_attr.flatten(0, 2),
                "residue_to_sampled_lig_triplet": aJ_grid_attr.flatten(0, 2),
                "sampled_lig_triplet_to_sampled_lig_triplet": IJ_grid_attr.flatten(
                    0, 2
                ),
            }
            edge_reps = graph_batcher.zero_pad_edge_attr(edge_reps, self.dim, device)
        else:
            graph_batcher = make_multi_relation_graph_batcher(
                self.graph_relations[:2], indexer, metadata
            )
            merged_edge_idx = graph_batcher.collate_idx_list(indexer)

            node_reps = {
                "prot_res": residue_rep,
            }
            edge_reps = {
                "residue_to_residue": rec_pair_rep,
                "sampled_residue_to_sampled_residue": AB_grid_attr_flat,
            }
            edge_reps = graph_batcher.zero_pad_edge_attr(edge_reps, self.dim, device)

        # Intertwine graph iterations and triangle iterations
        gather_idx_res_protpatch = indexer["gather_idx_a_pid"]
        for block_id in range(self.n_blocks):
            # Communicate between atomistic and patch resolutions
            # Up-sampling for interface edge embeddings
            rec_pair_rep = edge_reps["residue_to_residue"]
            AB_grid_attr_flat = edge_reps["sampled_residue_to_sampled_residue"]
            AB_grid_attr = AB_grid_attr_flat.view(
                batch_size,
                n_protein_patches,
                n_protein_patches,
                self.pair_dim,
            )

            if not batch["misc"]["protein_only"]:
                # Symmetrize off-diagonal blocks
                AJ_grid_attr_flat_ = (
                    edge_reps["sampled_residue_to_sampled_lig_triplet"]
                    + edge_reps["sampled_lig_triplet_to_sampled_residue"]
                ) / 2
                AJ_grid_attr = AJ_grid_attr_flat_.contiguous().view(
                    batch_size, n_protein_patches, n_ligand_patches, -1
                )
                aJ_grid_attr_flat_ = (
                    edge_reps["residue_to_sampled_lig_triplet"]
                    + edge_reps["sampled_lig_triplet_to_residue"]
                ) / 2
                aJ_grid_attr = aJ_grid_attr_flat_.contiguous().view(
                    batch_size, n_a_per_sample, n_ligand_patches, -1
                )
                IJ_grid_attr = (
                    edge_reps["sampled_lig_triplet_to_sampled_lig_triplet"]
                    .contiguous()
                    .view(batch_size, n_ligand_patches, n_ligand_patches, -1)
                )
                AJ_grid_attr_temp_, aJ_grid_attr_temp_ = self.AaJ_mha(
                    AJ_grid_attr.flatten(0, 1),
                    aJ_grid_attr.flatten(0, 1),
                    (
                        gather_idx_res_protpatch,
                        torch.arange(gather_idx_res_protpatch.shape[0], device=device),
                    ),
                )
                AJ_grid_attr = AJ_grid_attr_temp_.contiguous().view(
                    batch_size, n_protein_patches, n_ligand_patches, -1
                )
                aJ_grid_attr = aJ_grid_attr_temp_.contiguous().view(
                    batch_size, n_a_per_sample, n_ligand_patches, -1
                )
                merged_grid_rep = torch.cat(
                    [
                        torch.cat([AB_grid_attr, AJ_grid_attr], dim=2),
                        torch.cat([AJ_grid_attr.transpose(1, 2), IJ_grid_attr], dim=2),
                    ],
                    dim=1,
                )
            else:
                merged_grid_rep = AB_grid_attr

            # Inter-patch triangle attentions
            _, merged_grid_rep = self.triangle_stacks[block_id](
                merged_grid_rep,
                merged_grid_rep,
                merged_grid_rep.unsqueeze(-4),
            )

            # Dis-assemble the grid representation
            AB_grid_attr = merged_grid_rep[:, :n_protein_patches, :n_protein_patches]
            # Transfer grid-formatted representations to edges
            edge_reps["residue_to_residue"] = rec_pair_rep
            edge_reps["sampled_residue_to_sampled_residue"] = AB_grid_attr.flatten(0, 2)

            if not batch["misc"]["protein_only"]:
                AJ_grid_attr = merged_grid_rep[
                    :, :n_protein_patches, n_protein_patches:
                ].contiguous()
                IJ_grid_attr = merged_grid_rep[
                    :, n_protein_patches:, n_protein_patches:
                ].contiguous()

                edge_reps[
                    "sampled_residue_to_sampled_lig_triplet"
                ] = AJ_grid_attr.flatten(0, 2)
                edge_reps[
                    "sampled_lig_triplet_to_sampled_residue"
                ] = AJ_grid_attr.flatten(0, 2)
                edge_reps["residue_to_sampled_lig_triplet"] = aJ_grid_attr.flatten(0, 2)
                edge_reps["sampled_lig_triplet_to_residue"] = aJ_grid_attr.flatten(0, 2)
                edge_reps[
                    "sampled_lig_triplet_to_sampled_lig_triplet"
                ] = IJ_grid_attr.flatten(0, 2)
            merged_node_reps = graph_batcher.collate_node_attr(node_reps)
            merged_edge_reps = graph_batcher.collate_edge_attr(edge_reps)

            # Graph transformer iteration
            _, merged_node_reps, merged_edge_reps = self.graph_stacks[block_id](
                merged_node_reps,
                merged_node_reps,
                merged_edge_idx,
                merged_edge_reps,
            )
            node_reps = graph_batcher.offload_node_attr(merged_node_reps)
            edge_reps = graph_batcher.offload_edge_attr(merged_edge_reps)

        batch["features"][f"rec_res_attr{out_attr_suffix}"] = node_reps["prot_res"]
        batch["features"][f"res_res_pair_attr{out_attr_suffix}"] = edge_reps[
            "residue_to_residue"
        ]
        batch["features"][f"res_res_grid_attr_flat{out_attr_suffix}"] = edge_reps[
            "sampled_residue_to_sampled_residue"
        ]
        if not batch["misc"]["protein_only"]:
            batch["features"][f"lig_trp_attr{out_attr_suffix}"] = node_reps["lig_trp"]
            batch["features"][f"res_trp_grid_attr_flat{out_attr_suffix}"] = edge_reps[
                "sampled_residue_to_sampled_lig_triplet"
            ]
            batch["features"][f"res_trp_pair_attr_flat{out_attr_suffix}"] = edge_reps[
                "residue_to_sampled_lig_triplet"
            ]
            batch["features"][f"trp_trp_grid_attr_flat{out_attr_suffix}"] = edge_reps[
                "sampled_lig_triplet_to_sampled_lig_triplet"
            ]
            batch["metadata"]["n_lig_patches_per_sample"] = n_ligand_patches
        return batch
