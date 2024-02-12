import torch
from torch import Tensor, nn

from neuralplexer.model.common import GELUMLP, segment_mean
from neuralplexer.model.embedding import (GaussianFourierEncoding1D,
                                          RelativeGeometryEncoding)
from neuralplexer.model.modules import PointSetAttention
from neuralplexer.util.frame import RigidTransform, get_frame_matrix
from neuralplexer.util.tensorgraph import make_multi_relation_graph_batcher


class LocalUpdateUsingReferenceRotations(nn.Module):
    def __init__(
        self,
        fiber_dim,
        extra_feat_dim=0,
        eps=1e-4,
        dropout=0.0,
        hidden_dim=None,
        zero_init=False,
    ):
        super(LocalUpdateUsingReferenceRotations, self).__init__()
        self.dim = fiber_dim * 5 + extra_feat_dim
        self.fiber_dim = fiber_dim
        self.mlp = GELUMLP(
            self.dim,
            fiber_dim * 4,
            dropout=dropout,
            zero_init=zero_init,
            n_hidden_feats=hidden_dim,
        )
        self.eps = eps

    def forward(
        self,
        x: Tensor,
        rotation_mats: Tensor,
        extra_feats=None,
    ):
        # Vector norms are evaluated without applying rigid transform
        vecx_local = torch.matmul(
            x[:, 1:].transpose(-2, -1),
            rotation_mats,
        )
        x1_local = torch.cat(
            [
                x[:, 0],
                vecx_local.flatten(-2, -1),
                x[:, 1:].square().sum(dim=-2).add(self.eps).sqrt(),
            ],
            dim=-1,
        )
        if extra_feats is not None:
            x1_local = torch.cat([x1_local, extra_feats], dim=-1)
        x1_local = self.mlp(x1_local).view(-1, 4, self.fiber_dim)
        vecx1_out = torch.matmul(
            rotation_mats,
            x1_local[:, 1:],
        )
        x1_out = torch.cat([x1_local[:, :1], vecx1_out], dim=-2)
        return x1_out


class LocalUpdateUsingChannelWiseGating(nn.Module):
    def __init__(
        self, fiber_dim, eps=1e-4, dropout=0.0, hidden_dim=None, zero_init=False
    ):
        super(LocalUpdateUsingChannelWiseGating, self).__init__()
        self.dim = fiber_dim * 2
        self.fiber_dim = fiber_dim
        self.mlp = GELUMLP(
            self.dim,
            self.dim,
            dropout=dropout,
            n_hidden_feats=hidden_dim,
            zero_init=zero_init,
        )
        self.gate = nn.Sigmoid()
        self.lin_out = nn.Linear(fiber_dim, fiber_dim, bias=False)
        if zero_init:
            self.lin_out.weight.data.fill_(0.0)
        self.eps = eps

    def forward(
        self,
        x: Tensor,
    ):
        x1 = torch.cat(
            [
                x[:, 0],
                x[:, 1:].square().sum(dim=-2).add(self.eps).sqrt(),
            ],
            dim=-1,
        )
        x1 = self.mlp(x1)
        # Gated nonlinear operation on l=1 representations
        x1_scalar, x1_gatein = torch.split(x1, self.fiber_dim, dim=-1)
        x1_gate = self.gate(x1_gatein).unsqueeze(-2)
        vecx1_out = self.lin_out(x[:, 1:]).mul(x1_gate)
        x1_out = torch.cat([x1_scalar.unsqueeze(-2), vecx1_out], dim=-2)
        return x1_out


class EquivariantTransformerBlock(nn.Module):
    def __init__(
        self,
        fiber_dim,
        heads=8,
        point_dim=4,
        eps=1e-4,
        edge_dim=None,
        target_frames=False,
        edge_update=False,
        dropout=0.0,
    ):
        super(EquivariantTransformerBlock, self).__init__()
        self.attn_conv = PointSetAttention(
            fiber_dim,
            heads=heads,
            point_dim=point_dim,
            edge_dim=edge_dim,
            edge_update=edge_update,
        )
        self.fiber_dim = fiber_dim
        self.target_frames = target_frames
        self.eps = eps
        self.edge_update = edge_update
        if target_frames:
            self.local_update = LocalUpdateUsingReferenceRotations(
                fiber_dim, eps=eps, dropout=dropout, zero_init=True
            )
        else:
            self.local_update = LocalUpdateUsingChannelWiseGating(
                fiber_dim, eps=eps, dropout=dropout, zero_init=True
            )

    def forward(
        self,
        x: Tensor,
        edge_index: torch.LongTensor,
        t: Tensor,
        R: Tensor = None,
        x_edge: Tensor = None,
    ):
        if self.edge_update:
            xout, edge_out = self.attn_conv(x, x, edge_index, t, t, x_edge=x_edge)
            x_edge = x_edge + edge_out
        else:
            xout = self.attn_conv(x, x, edge_index, t, t, x_edge=x_edge)
        x = x + xout
        if self.target_frames:
            x = self.local_update(x, R) + x
        else:
            x = self.local_update(x) + x
        return x, x_edge


class EquivariantStructureDenoisingModule(nn.Module):
    def __init__(
        self,
        fiber_dim,
        input_dim,
        input_pair_dim,
        hidden_dim=1024,
        n_stacks=4,
        n_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_pair_dim = input_pair_dim
        self.fiber_dim = fiber_dim
        self.protatm_padding_dim = 37
        self.n_blocks = n_stacks
        self.input_node_projector = nn.Linear(input_dim, fiber_dim, bias=False)
        self.input_node_vec_projector = nn.Linear(input_dim, fiber_dim * 3, bias=False)
        self.input_pair_projector = nn.Linear(input_pair_dim, fiber_dim, bias=False)
        # Inherit the residue representations
        self.atm_embed = GELUMLP(input_dim + 32, fiber_dim)
        self.ipa_modules = nn.ModuleList(
            [
                EquivariantTransformerBlock(
                    fiber_dim,
                    heads=n_heads,
                    point_dim=fiber_dim // (n_heads * 2),
                    edge_dim=fiber_dim,
                    target_frames=True,
                    edge_update=True,
                    dropout=dropout,
                )
                for _ in range(n_stacks)
            ]
        )
        self.res_adapters = nn.ModuleList(
            [
                LocalUpdateUsingReferenceRotations(
                    fiber_dim,
                    extra_feat_dim=input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    zero_init=True,
                )
                for _ in range(n_stacks)
            ]
        )
        self.protatm_type_encoding = GELUMLP(
            self.protatm_padding_dim + input_dim, input_pair_dim
        )
        self.time_encoding = GaussianFourierEncoding1D(16)
        self.rel_geom_enc = RelativeGeometryEncoding(15, fiber_dim)
        self.rel_geom_embed = GELUMLP(fiber_dim, fiber_dim, n_hidden_feats=fiber_dim)
        # [displacement, scale]
        self.out_drift_res = nn.ModuleList(
            [nn.Linear(fiber_dim, 1, bias=False) for _ in range(n_stacks)]
        )
        # for i in range(n_stacks):
        #     self.out_drift_res[i].weight.data.fill_(0.0)
        self.out_scale_res = nn.ModuleList(
            [GELUMLP(fiber_dim, 1, zero_init=True) for _ in range(n_stacks)]
        )
        self.out_drift_atm = nn.ModuleList(
            [nn.Linear(fiber_dim, 1, bias=False) for _ in range(n_stacks)]
        )
        # for i in range(n_stacks):
        #     self.out_drift_atm[i].weight.data.fill_(0.0)
        self.out_scale_atm = nn.ModuleList(
            [GELUMLP(fiber_dim, 1, zero_init=True) for _ in range(n_stacks)]
        )

        # Pre-tabulated edges
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
                "prot_atm_to_prot_atm_graph",
                "protatm_protatm_idx_src",
                "protatm_protatm_idx_dst",
                "prot_atm",
                "prot_atm",
            ),
            (
                "prot_atm_to_prot_atm_knn",
                "knn_idx_protatm_protatm_src",
                "knn_idx_protatm_protatm_dst",
                "prot_atm",
                "prot_atm",
            ),
            (
                "prot_atm_to_residue",
                "protatm_res_idx_protatm",
                "protatm_res_idx_res",
                "prot_atm",
                "prot_res",
            ),
            (
                "residue_to_prot_atm",
                "protatm_res_idx_res",
                "protatm_res_idx_protatm",
                "prot_res",
                "prot_atm",
            ),
            (
                "sampled_lig_triplet_to_lig_atm",
                "gather_idx_UI_I",
                "gather_idx_UI_u",
                "lig_trp",
                "lig_atm",
            ),
            (
                "lig_atm_to_sampled_lig_triplet",
                "gather_idx_UI_u",
                "gather_idx_UI_I",
                "lig_atm",
                "lig_trp",
            ),
            (
                "lig_atm_to_lig_atm_graph",
                "gather_idx_uv_u",
                "gather_idx_uv_v",
                "lig_atm",
                "lig_atm",
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
            (
                "prot_atm_to_lig_atm_knn",
                "knn_idx_protatm_ligatm_src",
                "knn_idx_protatm_ligatm_dst",
                "prot_atm",
                "lig_atm",
            ),
            (
                "lig_atm_to_prot_atm_knn",
                "knn_idx_ligatm_protatm_src",
                "knn_idx_ligatm_protatm_dst",
                "lig_atm",
                "prot_atm",
            ),
            (
                "lig_atm_to_lig_atm_knn",
                "knn_idx_ligatm_ligatm_src",
                "knn_idx_ligatm_ligatm_dst",
                "lig_atm",
                "lig_atm",
            ),
        ]
        self.graph_relations_no_ligand = self.graph_relations[:6]

    def _init_scalar_vec_rep(self, x, x_v=None, frame=None):
        if frame is None:
            # Zero-initialize the vector channels
            vec_shape = (*x.shape[:-1], 3, x.shape[-1])
            res = torch.cat(
                [x.unsqueeze(-2), torch.zeros(vec_shape, device=x.device)], dim=-2
            )
        else:
            x_v = x_v.view(*x.shape[:-1], 3, x.shape[-1])
            x_v_glob = torch.matmul(frame.R, x_v)
            res = torch.cat([x.unsqueeze(-2), x_v_glob], dim=-2)
        return res

    def forward(
        self,
        batch,
        frozen_prot=False,
        **kwargs,
    ):
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        metadata["num_structid"]
        max(metadata["num_a_per_sample"])

        prot_res_rep_in = features["rec_res_attr_decin"]
        timestep_prot = features["timestep_encoding_prot"]
        device = prot_res_rep_in.device

        # Protein all-atom representation initialization
        protatm_padding_mask = features["res_atom_mask"]
        protatm_atom37_onehot = nn.functional.one_hot(
            features["protatm_to_atom37_index"], num_classes=self.protatm_padding_dim
        )
        protatm_res_pair_encoding = self.protatm_type_encoding(
            torch.cat(
                [
                    prot_res_rep_in[indexer["protatm_res_idx_res"]],
                    protatm_atom37_onehot,
                ],
                dim=-1,
            )
        )
        # Gathered AA features from individual graphs
        prot_atm_rep_in = features["prot_atom_attr_projected"]
        prot_atm_rep_int = self.atm_embed(
            torch.cat(
                [
                    prot_atm_rep_in,
                    self.time_encoding(timestep_prot)[indexer["protatm_res_idx_res"]],
                ],
                dim=-1,
            )
        )

        prot_atm_coords_padded = features["input_protein_coords"]
        prot_atm_coords_flat = prot_atm_coords_padded[protatm_padding_mask]

        # Embed the rigid body node representations
        backbone_frames = get_frame_matrix(
            prot_atm_coords_padded[:, 0],
            prot_atm_coords_padded[:, 1],
            prot_atm_coords_padded[:, 2],
        )
        prot_res_rep = self._init_scalar_vec_rep(
            self.input_node_projector(prot_res_rep_in),
            x_v=self.input_node_vec_projector(prot_res_rep_in),
            frame=backbone_frames,
        )
        prot_atm_rep = self._init_scalar_vec_rep(prot_atm_rep_int)
        # gather AA features from individual graphs
        node_reps = {
            "prot_res": prot_res_rep,
            "prot_atm": prot_atm_rep,
        }
        # Embed pair representations
        edge_reps = {
            "residue_to_residue": features["res_res_pair_attr_decin"],
            "prot_atm_to_prot_atm_graph": features["prot_atom_pair_attr_projected"],
            "prot_atm_to_prot_atm_knn": features["knn_feat_protatm_protatm"],
            "prot_atm_to_residue": protatm_res_pair_encoding,
            "residue_to_prot_atm": protatm_res_pair_encoding,
            "sampled_residue_to_sampled_residue": features[
                "res_res_grid_attr_flat_decin"
            ],
        }

        if not batch["misc"]["protein_only"]:
            max(metadata["num_i_per_sample"])
            timestep_lig = features["timestep_encoding_lig"]
            lig_atm_rep_in = features["lig_atom_attr_projected"]
            lig_frame_rep_in = features["lig_trp_attr_decin"]
            # Ligand atom embedding. Two timescales
            lig_atm_rep_int = self.atm_embed(
                torch.cat(
                    [lig_atm_rep_in, self.time_encoding(timestep_lig)],
                    dim=-1,
                )
            )
            lig_atm_rep = self._init_scalar_vec_rep(lig_atm_rep_int)

            # Prepare ligand atom - sidechain atom indexers
            # Initialize coordinate features
            lig_atm_coords = features["input_ligand_coords"].clone()

            lig_frame_atm_idx = (
                indexer["gather_idx_ijk_i"][indexer["gather_idx_I_ijk"]],
                indexer["gather_idx_ijk_j"][indexer["gather_idx_I_ijk"]],
                indexer["gather_idx_ijk_k"][indexer["gather_idx_I_ijk"]],
            )
            ligand_trp_frames = get_frame_matrix(
                lig_atm_coords[lig_frame_atm_idx[0]],
                lig_atm_coords[lig_frame_atm_idx[1]],
                lig_atm_coords[lig_frame_atm_idx[2]],
            )
            lig_frame_rep = self._init_scalar_vec_rep(
                self.input_node_projector(lig_frame_rep_in),
                x_v=self.input_node_vec_projector(lig_frame_rep_in),
                frame=ligand_trp_frames,
            )
            node_reps.update(
                {
                    "lig_atm": lig_atm_rep,
                    "lig_trp": lig_frame_rep,
                }
            )
            edge_reps.update(
                {
                    "lig_atm_to_lig_atm_graph": features[
                        "lig_atom_pair_attr_projected"
                    ],
                    "sampled_lig_triplet_to_lig_atm": features[
                        "lig_af_pair_attr_projected"
                    ],
                    "lig_atm_to_sampled_lig_triplet": features[
                        "lig_af_pair_attr_projected"
                    ],
                    "sampled_residue_to_sampled_lig_triplet": features[
                        "res_trp_grid_attr_flat_decin"
                    ],
                    "sampled_lig_triplet_to_sampled_residue": features[
                        "res_trp_grid_attr_flat_decin"
                    ],
                    "residue_to_sampled_lig_triplet": features[
                        "res_trp_pair_attr_flat_decin"
                    ],
                    "sampled_lig_triplet_to_residue": features[
                        "res_trp_pair_attr_flat_decin"
                    ],
                    "sampled_lig_triplet_to_sampled_lig_triplet": features[
                        "trp_trp_grid_attr_flat_decin"
                    ],
                    "prot_atm_to_lig_atm_knn": features["knn_feat_protatm_ligatm"],
                    "lig_atm_to_prot_atm_knn": features["knn_feat_ligatm_protatm"],
                    "lig_atm_to_lig_atm_knn": features["knn_feat_ligatm_ligatm"],
                }
            )

        # Message passing
        protatm_res_idx_res = indexer["protatm_res_idx_res"]
        if batch["misc"]["protein_only"]:
            graph_relations = self.graph_relations_no_ligand
        else:
            graph_relations = self.graph_relations
        graph_batcher = make_multi_relation_graph_batcher(
            graph_relations, indexer, metadata
        )
        merged_edge_idx = graph_batcher.collate_idx_list(indexer)
        merged_node_reps = graph_batcher.collate_node_attr(node_reps)
        merged_edge_reps = graph_batcher.collate_edge_attr(
            graph_batcher.zero_pad_edge_attr(edge_reps, self.input_pair_dim, device)
        )
        merged_edge_reps = self.input_pair_projector(merged_edge_reps)
        assert merged_edge_idx[0].shape[0] == merged_edge_reps.shape[0]
        assert merged_edge_idx[1].shape[0] == merged_edge_reps.shape[0]

        dummy_prot_atm_frames = RigidTransform(prot_atm_coords_flat, R=None)
        if not batch["misc"]["protein_only"]:
            dummy_lig_atm_frames = RigidTransform(lig_atm_coords, R=None)
            merged_node_t = graph_batcher.collate_node_attr(
                {
                    "prot_res": backbone_frames.t,
                    "prot_atm": dummy_prot_atm_frames.t,
                    "lig_atm": dummy_lig_atm_frames.t,
                    "lig_trp": ligand_trp_frames.t,
                }
            )
            merged_node_R = graph_batcher.collate_node_attr(
                {
                    "prot_res": backbone_frames.R,
                    "prot_atm": dummy_prot_atm_frames.R,
                    "lig_atm": dummy_lig_atm_frames.R,
                    "lig_trp": ligand_trp_frames.R,
                }
            )
        else:
            merged_node_t = graph_batcher.collate_node_attr(
                {
                    "prot_res": backbone_frames.t,
                    "prot_atm": dummy_prot_atm_frames.t,
                }
            )
            merged_node_R = graph_batcher.collate_node_attr(
                {
                    "prot_res": backbone_frames.R,
                    "prot_atm": dummy_prot_atm_frames.R,
                }
            )
        merged_node_frames = RigidTransform(merged_node_t, merged_node_R)
        merged_edge_reps = merged_edge_reps + (
            self.rel_geom_embed(
                self.rel_geom_enc(merged_node_frames, merged_edge_idx)
                + merged_edge_reps
            )
        )

        # No need to reassign embeddings but need to update point coordinates & frames
        for block_id in range(self.n_blocks):
            dummy_prot_atm_frames = RigidTransform(prot_atm_coords_flat, R=None)
            if not batch["misc"]["protein_only"]:
                dummy_lig_atm_frames = RigidTransform(lig_atm_coords, R=None)
                merged_node_t = graph_batcher.collate_node_attr(
                    {
                        "prot_res": backbone_frames.t,
                        "prot_atm": dummy_prot_atm_frames.t,
                        "lig_atm": dummy_lig_atm_frames.t,
                        "lig_trp": ligand_trp_frames.t,
                    }
                )
                merged_node_R = graph_batcher.collate_node_attr(
                    {
                        "prot_res": backbone_frames.R,
                        "prot_atm": dummy_prot_atm_frames.R,
                        "lig_atm": dummy_lig_atm_frames.R,
                        "lig_trp": ligand_trp_frames.R,
                    }
                )
            else:
                merged_node_t = graph_batcher.collate_node_attr(
                    {
                        "prot_res": backbone_frames.t,
                        "prot_atm": dummy_prot_atm_frames.t,
                    }
                )
                merged_node_R = graph_batcher.collate_node_attr(
                    {
                        "prot_res": backbone_frames.R,
                        "prot_atm": dummy_prot_atm_frames.R,
                    }
                )
            # PredictDrift iteration
            merged_node_reps, merged_edge_reps = self.ipa_modules[block_id](
                merged_node_reps,
                merged_edge_idx,
                t=merged_node_t,
                R=merged_node_R,
                x_edge=merged_edge_reps,
            )
            offloaded_node_reps = graph_batcher.offload_node_attr(merged_node_reps)
            if "lig_trp" in offloaded_node_reps.keys():
                lig_frame_rep = offloaded_node_reps["lig_trp"]
                offloaded_node_reps["lig_trp"] = lig_frame_rep + self.res_adapters[
                    block_id
                ](lig_frame_rep, ligand_trp_frames.R, extra_feats=lig_frame_rep_in)
            prot_res_rep = offloaded_node_reps["prot_res"]
            offloaded_node_reps["prot_res"] = prot_res_rep + self.res_adapters[
                block_id
            ](prot_res_rep, backbone_frames.R, extra_feats=prot_res_rep_in)
            merged_node_reps = graph_batcher.collate_node_attr(offloaded_node_reps)
            # Displacement vectors in the global coordinate system
            if not batch["misc"]["protein_only"]:
                drift_trp = (
                    self.out_drift_res[block_id](
                        offloaded_node_reps["lig_trp"][:, 1:]
                    ).squeeze(-1)
                    * torch.sigmoid(
                        self.out_scale_res[block_id](
                            offloaded_node_reps["lig_trp"][:, 0]
                        )
                    )
                    * 10
                )
                drift_trp_gathered = segment_mean(
                    drift_trp,
                    indexer["gather_idx_I_molid"],
                    metadata["num_molid"],
                )[indexer["gather_idx_i_molid"]]
                drift_atm = self.out_drift_atm[block_id](
                    offloaded_node_reps["lig_atm"][:, 1:]
                ).squeeze(-1) * torch.sigmoid(
                    self.out_scale_atm[block_id](offloaded_node_reps["lig_atm"][:, 0])
                )
                lig_atm_coords = lig_atm_coords + drift_atm + drift_trp_gathered
                ligand_trp_frames = get_frame_matrix(
                    lig_atm_coords[lig_frame_atm_idx[0]],
                    lig_atm_coords[lig_frame_atm_idx[1]],
                    lig_atm_coords[lig_frame_atm_idx[2]],
                )

            drift_bb = (
                self.out_drift_res[block_id](
                    offloaded_node_reps["prot_res"][:, 1:]
                ).squeeze(-1)
                * torch.sigmoid(
                    self.out_scale_res[block_id](offloaded_node_reps["prot_res"][:, 0])
                )
                * 10
            )
            drift_bb_gathered = drift_bb[protatm_res_idx_res]
            drift_prot_atm_int = self.out_drift_atm[block_id](
                offloaded_node_reps["prot_atm"][:, 1:]
            ).squeeze(-1) * torch.sigmoid(
                self.out_scale_atm[block_id](offloaded_node_reps["prot_atm"][:, 0])
            )
            if not frozen_prot:
                prot_atm_coords_flat = (
                    prot_atm_coords_flat + drift_prot_atm_int + drift_bb_gathered
                )

            prot_atm_coords_padded = torch.zeros_like(features["input_protein_coords"])
            prot_atm_coords_padded[protatm_padding_mask] = prot_atm_coords_flat
            backbone_frames = get_frame_matrix(
                prot_atm_coords_padded[:, 0],
                prot_atm_coords_padded[:, 1],
                prot_atm_coords_padded[:, 2],
            )

        ret = {
            "final_embedding_prot_atom": offloaded_node_reps["prot_atm"],
            "final_embedding_prot_res": offloaded_node_reps["prot_res"],
            "final_coords_prot_atom": prot_atm_coords_flat,
            "final_coords_prot_atom_padded": prot_atm_coords_padded,
        }
        if not batch["misc"]["protein_only"]:
            ret["final_embedding_lig_atom"] = offloaded_node_reps["lig_atm"]
            ret["final_coords_lig_atom"] = lig_atm_coords
        else:
            ret["final_embedding_lig_atom"] = None
            ret["final_coords_lig_atom"] = None
        return ret
