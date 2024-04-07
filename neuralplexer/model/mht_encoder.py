import math
import random
from abc import ABC
from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from neuralplexer.model.common import (GELUMLP, SumPooling, segment_mean,
                                       segment_softmax, segment_sum)
from neuralplexer.model.embedding import GaussianFourierEncoding1D
from neuralplexer.model.esdm import EquivariantTransformerBlock
from neuralplexer.model.modules import GlobalAttention, TransformerLayer
from neuralplexer.util.distributions import MixtureGPSNetwork3D
from neuralplexer.util.frame import (RigidTransform, cartesian_to_internal,
                                     get_frame_matrix)
from neuralplexer.util.tensorgraph import make_multi_relation_graph_batcher


class PathConvStack(nn.Module):
    def __init__(
        self,
        pair_channels,
        n_heads=8,
        max_pi_length=8,
        dropout=0.0,
    ):
        super(PathConvStack, self).__init__()
        self.pair_channels = pair_channels
        self.max_pi_length = max_pi_length
        self.n_heads = n_heads

        self.prop_value_layer = nn.Linear(pair_channels, n_heads, bias=False)
        self.triangle_pair_kernel_layer = nn.Linear(pair_channels, n_heads, bias=False)
        self.prop_update_mlp = GELUMLP(
            n_heads * (max_pi_length + 1), pair_channels, dropout=dropout
        )

    def forward(
        self,
        prop_attr: torch.Tensor,
        stereo_attr: torch.Tensor,
        indexer: Dict[str, torch.LongTensor],
        metadata,
    ):
        triangle_pair_kernel = self.triangle_pair_kernel_layer(stereo_attr)
        # Segment-wise softmax, normalized by outgoing triangles
        triangle_pair_alpha = segment_softmax(
            triangle_pair_kernel, indexer["gather_idx_ijkl_jkl"], metadata["num_ijk"]
        )  # .div(self.max_pi_length)
        # Uijk,ijkl->ujkl pair representation update
        kernel = triangle_pair_alpha[indexer["gather_idx_Uijkl_ijkl"]]
        out_prop_attr = [self.prop_value_layer(prop_attr)]
        for _ in range(self.max_pi_length):
            gathered_prop_attr = out_prop_attr[-1][indexer["gather_idx_Uijkl_Uijk"]]
            out_prop_attr.append(
                segment_sum(
                    kernel.mul(gathered_prop_attr),
                    indexer["gather_idx_Uijkl_ujkl"],
                    metadata["num_Uijk"],
                )
            )
        new_prop_attr = torch.cat(out_prop_attr, dim=-1)
        new_prop_attr = self.prop_update_mlp(new_prop_attr) + prop_attr
        return new_prop_attr


class PIFormer(nn.Module):
    def __init__(
        self,
        node_channels,
        pair_channels,
        n_atom_encodings,
        n_bond_encodings,
        n_atom_pos_encodings,
        n_stereo_encodings,
        heads,
        head_dim,
        max_path_length=4,
        n_transformer_stacks=4,
        hidden_dim=None,
        dropout=0.0,
    ):
        super(PIFormer, self).__init__()
        self.node_channels = node_channels
        self.pair_channels = pair_channels
        self.max_pi_length = max_path_length
        self.n_transformer_stacks = n_transformer_stacks
        self.n_atom_encodings = n_atom_encodings
        self.n_bond_encodings = n_bond_encodings
        self.n_atom_pair_encodings = n_bond_encodings + 4
        self.n_atom_pos_encodings = n_atom_pos_encodings

        self.input_atom_layer = nn.Linear(n_atom_encodings, node_channels)
        self.input_pair_layer = nn.Linear(self.n_atom_pair_encodings, pair_channels)
        self.input_stereo_layer = nn.Linear(n_stereo_encodings, pair_channels)
        self.input_prop_layer = GELUMLP(
            self.n_atom_pair_encodings * 3,
            pair_channels,
        )
        self.path_integral_stacks = nn.ModuleList(
            [
                PathConvStack(
                    pair_channels,
                    max_pi_length=max_path_length,
                    dropout=dropout,
                )
                for _ in range(n_transformer_stacks)
            ]
        )
        self.graph_transformer_stacks = nn.ModuleList(
            [
                TransformerLayer(
                    node_channels,
                    heads,
                    head_dim=head_dim,
                    edge_channels=pair_channels,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    edge_update=True,
                )
                for _ in range(n_transformer_stacks)
            ]
        )

    def forward(self, batch: Dict, masking_rate=0):
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]
        features["atom_encodings"] = features["atom_encodings"]
        atom_attr = features["atom_encodings"]
        atom_pair_attr = features["atom_pair_encodings"]
        af_pair_attr = features["atom_frame_pair_encodings"]
        stereo_enc = features["stereo_chemistry_encodings"]
        batch["features"]["lig_atom_token"] = atom_attr.detach().clone()
        batch["features"]["lig_pair_token"] = atom_pair_attr.detach().clone()

        atom_mask = (
            torch.rand(atom_attr.shape[0], device=atom_attr.device) > masking_rate
        )
        stereo_mask = (
            torch.rand(stereo_enc.shape[0], device=stereo_enc.device) > masking_rate
        )
        atom_pair_mask = (
            torch.rand(atom_pair_attr.shape[0], device=atom_pair_attr.device)
            > masking_rate
        )
        af_pair_mask = (
            torch.rand(af_pair_attr.shape[0], device=atom_pair_attr.device)
            > masking_rate
        )
        atom_attr = atom_attr * atom_mask[:, None]
        stereo_enc = stereo_enc * stereo_mask[:, None]
        atom_pair_attr = atom_pair_attr * atom_pair_mask[:, None]
        af_pair_attr = af_pair_attr * af_pair_mask[:, None]

        # Embedding blocks
        metadata["num_atom"] = metadata["num_u"]
        metadata["num_frame"] = metadata["num_ijk"]
        atom_attr = self.input_atom_layer(atom_attr)
        atom_pair_attr = self.input_pair_layer(atom_pair_attr)
        triangle_attr = atom_attr.new_zeros(metadata["num_frame"], self.node_channels)
        # Initialize atom-frame pair attributes. Reusing uv indices
        prop_attr = self.input_prop_layer(af_pair_attr)
        stereo_attr = self.input_stereo_layer(stereo_enc)

        graph_relations = [
            ("atom_to_atom", "gather_idx_uv_u", "gather_idx_uv_v", "atom", "atom"),
            (
                "atom_to_frame",
                "gather_idx_Uijk_u",
                "gather_idx_Uijk_ijk",
                "atom",
                "frame",
            ),
            (
                "frame_to_atom",
                "gather_idx_Uijk_ijk",
                "gather_idx_Uijk_u",
                "frame",
                "atom",
            ),
            (
                "frame_to_frame",
                "gather_idx_ijkl_ijk",
                "gather_idx_ijkl_jkl",
                "frame",
                "frame",
            ),
        ]

        graph_batcher = make_multi_relation_graph_batcher(
            graph_relations, indexer, metadata
        )
        merged_edge_idx = graph_batcher.collate_idx_list(indexer)
        node_reps = {"atom": atom_attr, "frame": triangle_attr}
        edge_reps = {
            "atom_to_atom": atom_pair_attr,
            "atom_to_frame": prop_attr,
            "frame_to_atom": prop_attr,
            "frame_to_frame": stereo_attr,
        }

        # Graph path integral recursion
        for block_id in range(self.n_transformer_stacks):
            merged_node_attr = graph_batcher.collate_node_attr(node_reps)
            merged_edge_attr = graph_batcher.collate_edge_attr(edge_reps)
            _, merged_node_attr, merged_edge_attr = self.graph_transformer_stacks[
                block_id
            ](
                merged_node_attr,
                merged_node_attr,
                merged_edge_idx,
                merged_edge_attr,
            )
            node_reps = graph_batcher.offload_node_attr(merged_node_attr)
            edge_reps = graph_batcher.offload_edge_attr(merged_edge_attr)
            prop_attr = edge_reps["atom_to_frame"]
            stereo_attr = edge_reps["frame_to_frame"]
            prop_attr = prop_attr + self.path_integral_stacks[block_id](
                prop_attr,
                stereo_attr,
                indexer,
                metadata,
            )
            edge_reps["atom_to_frame"] = prop_attr

        node_reps["sampled_frame"] = node_reps["frame"][indexer["gather_idx_I_ijk"]]

        batch["metadata"]["num_lig_atm"] = metadata["num_u"]
        batch["metadata"]["num_lig_trp"] = metadata["num_I"]

        batch["features"]["lig_atom_attr"] = node_reps["atom"]
        # Downsampled ligand frames
        batch["features"]["lig_trp_attr"] = node_reps["sampled_frame"]
        batch["features"]["lig_atom_pair_attr"] = edge_reps["atom_to_atom"]
        batch["features"]["lig_prop_attr"] = edge_reps["atom_to_frame"]
        edge_reps["sampled_atom_to_sampled_frame"] = edge_reps["atom_to_frame"][
            indexer["gather_idx_UI_Uijk"]
        ]
        batch["features"]["lig_af_pair_attr"] = edge_reps[
            "sampled_atom_to_sampled_frame"
        ]
        return batch


def _resolve_ligand_encoder(ligand_model_config, task_config):
    model = PIFormer(
        ligand_model_config.node_channels,
        ligand_model_config.pair_channels,
        ligand_model_config.n_atom_encodings,
        ligand_model_config.n_bond_encodings,
        ligand_model_config.n_atom_pos_encodings,
        ligand_model_config.n_stereo_encodings,
        ligand_model_config.n_attention_heads,
        ligand_model_config.attention_head_dim,
        hidden_dim=ligand_model_config.hidden_dim,
        max_path_length=ligand_model_config.max_path_integral_length,
        n_transformer_stacks=ligand_model_config.n_transformer_stacks,
        dropout=task_config.dropout,
    )
    if ligand_model_config.from_pretrained:
        try:
            pretrained_dict = {
                ".".join(k.split(".")[1:]): v
                for k, v in torch.load(ligand_model_config.checkpoint_file)[
                    "state_dict"
                ].items()
                if k.startswith("ligand_encoder")
            }
            model.load_state_dict(pretrained_dict)
        except:
            print("Could not load pretrained MHT weights, skipping")
    return model


class MolPropRegressor(pl.LightningModule, ABC):
    def __init__(self, config):
        super(MolPropRegressor, self).__init__()
        self.config = config.mol_encoder
        self.global_config = config.task
        self.encoder_stack = _resolve_ligand_encoder(self.config, self.global_config)
        self.task_head = nn.Sequential(
            GELUMLP(self.config.node_channels, self.config.node_channels),
            nn.Linear(self.config.node_channels, 1),
        )
        self.task_head[1].weight.data.fill_(0.0)
        self.task_head[1].bias.data.fill_(0.0)
        self.pooling = SumPooling()

    def forward(self, metadata, indexer, features):
        atom_attr, _, _ = self.encoder_stack(
            features["atom_encodings"],
            features["bond_encodings"],
            features["atom_positional_encodings"],
            features["stereo_chemistry_encodings"],
            indexer,
        )
        out = self.pooling(
            self.task_head(atom_attr),
            indexer["gather_idx_i_molid"],
            metadata["num_molid"],
        )
        return out.squeeze(-1)

    def training_step(self, batch, batch_idx):
        self.train()
        out = self.forward(batch["metadata"], batch["indexer"], batch["features"])
        loss = F.mse_loss(out, batch["labels"])
        self.log("train_loss", loss, on_epoch=True, batch_size=batch["batch_size"])
        return {"loss": loss, "train_loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch["metadata"], batch["indexer"], batch["features"])
        loss = F.mse_loss(out, batch["labels"], reduction="none")
        return loss

    def validation_epoch_end(self, outs):
        mse = torch.cat(outs, dim=0).mean()
        self.log("val_loss", mse)
        self.log("val_rmse", mse.sqrt())

    def test_step(self, batch, batch_idx):
        self.eval()
        out = self.forward(batch["metadata"], batch["indexer"], batch["features"])
        loss = F.mse_loss(out, batch["labels"], reduction="none")
        return loss

    def test_epoch_end(self, outs):
        mse = torch.cat(outs, dim=0).mean()
        self.log("test_loss", mse)
        self.log("test_rmse", mse.sqrt())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.global_config.init_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class MolPretrainingWrapper(pl.LightningModule, ABC):
    def __init__(self, config):
        super(MolPretrainingWrapper, self).__init__()
        self.config = config.mol_encoder
        self.task_config = config.task
        assert self.config.model_name == "piformer"
        self.encoder_stack = _resolve_ligand_encoder(self.config, self.task_config)
        self.R_ref = 10
        self.task_head = MixtureGPSNetwork3D(
            self.config.pair_channels,
            self.task_config.n_modes,
            r_0=self.R_ref,
        )
        self.score_head = MolGeomPredictionHead(
            self.config.node_channels,
            self.config.pair_channels,
            dropout=self.task_config.dropout,
        )
        self.global_pooler = GlobalAttention(
            self.config.node_channels,
            self.task_config.global_heads,
            self.task_config.global_head_dim,
        )
        self.global_readout = GELUMLP(
            self.task_config.global_heads * self.task_config.global_head_dim,
            self.task_config.label_dim * 2,
            n_hidden_feats=self.task_config.global_heads
            * self.task_config.global_head_dim,
            dropout=self.task_config.dropout,
        )
        self.mlm_head = GELUMLP(
            self.config.node_channels,
            self.config.n_atom_encodings,
            n_hidden_feats=self.config.node_channels,
        )
        self.mlm_pair_head = GELUMLP(
            self.config.pair_channels,
            self.config.n_bond_encodings + 4,
            n_hidden_feats=self.config.node_channels,
        )
        self.regression_metric = nn.MSELoss()
        self.classification_metric = nn.CrossEntropyLoss()
        self.aux_metric = nn.BCEWithLogitsLoss()
        self.max_masking_rate = self.task_config.max_masking_rate
        self.save_hyperparameters()

    @staticmethod
    def compute_internal_coord_samples(xyz, indexer, prop_idx=None):
        triplet_frames = get_frame_matrix(
            xyz[indexer["gather_idx_ijk_i"], :],
            xyz[indexer["gather_idx_ijk_j"], :],
            xyz[indexer["gather_idx_ijk_k"], :],
        )
        if prop_idx is not None:
            gathered_xyz = torch.index_select(
                xyz, dim=0, index=indexer["gather_idx_Uijk_u"][prop_idx]
            )
            triplet_frames_uijk = triplet_frames[
                indexer["gather_idx_Uijk_ijk"][prop_idx]
            ]
        else:
            gathered_xyz = torch.index_select(
                xyz, dim=0, index=indexer["gather_idx_Uijk_u"]
            )
            triplet_frames_uijk = triplet_frames[indexer["gather_idx_Uijk_ijk"]]
        internal_frame_observations = cartesian_to_internal(
            gathered_xyz, triplet_frames_uijk
        )
        # Assemble into an extra batch dim
        return internal_frame_observations.transpose(0, 1)

    def forward(self, batch):
        masking_rate = random.uniform(0, self.max_masking_rate)
        batch = self.encoder_stack(batch, masking_rate=masking_rate)
        return batch

    def compute_supervised_loss(self, batch):
        indexer = batch["indexer"]
        features = batch["features"]
        index_combined = torch.cat(
            [indexer["gather_idx_I_molid"], indexer["gather_idx_i_molid"]], dim=0
        )
        features_combined = torch.cat(
            [features["lig_trp_attr"], features["lig_atom_attr"]], dim=0
        )
        pooled_embedding = self.global_pooler(
            features_combined, index_combined, batch["metadata"]["num_molid"]
        )
        global_pred = self.global_readout(pooled_embedding).view(
            -1, self.task_config.label_dim, 2
        )
        label_mask = batch["labels"]["chemicalchecker25_sign1_mask"].bool()
        regression_loss = self.regression_metric(
            global_pred[:, :, 0][label_mask],
            batch["labels"]["chemicalchecker25_sign1_value"][label_mask],
        ) / torch.std(
            batch["labels"]["chemicalchecker25_sign1_value"][label_mask]
        ).square().add(
            1e-8
        )
        aux_loss = self.aux_metric(global_pred[:, :, 1], label_mask.float())
        return regression_loss, aux_loss

    def compute_mlm_loss(self, batch):
        # Masked language modelling
        features = batch["features"]
        atom_token_pred = self.mlm_head(features["lig_atom_attr"])
        mlm_loss_atom = self.classification_metric(
            atom_token_pred, batch["features"]["lig_atom_token"]
        )
        pair_token_pred = self.mlm_pair_head(features["lig_atom_pair_attr"])
        mlm_loss_pair = self.classification_metric(
            pair_token_pred, batch["features"]["lig_pair_token"]
        )
        return mlm_loss_atom, mlm_loss_pair

    def _get_noise_schedule(self, t, T, name="linear"):
        if name == "cosine":
            # Cosine schedule
            alpha_t = torch.cos((t / T + 0.05) / 1.05 * math.pi / 2) ** 2
            alpha_tm = torch.cos(((t - 1) / T + 0.05) / 1.05 * math.pi / 2) ** 2
        elif name == "linear":
            # Linear schedule
            alpha_t = torch.exp(-t / T - 5 * (t / T) ** 2)
            alpha_tm = torch.exp(-(t - 1) / T - 5 * ((t - 1) / T) ** 2)
        else:
            raise NotImplementedError
        beta_t = 1 - alpha_t / alpha_tm
        return alpha_t, alpha_tm, beta_t

    def _forward_diffuse_alpha_t(self, x0, prior_z, alpha_t):
        xt = x0 * alpha_t.sqrt() + prior_z * (1 - alpha_t).sqrt()
        return xt

    def compute_neg_log_likelihood(self, batch, observations):
        out_dist = self.forward(batch)
        nll = -out_dist.log_prob(observations)
        return nll

    def compute_DDPM_loss(self, batch):
        features = batch["features"]
        indexer = batch["indexer"]
        T = 1000
        t = (
            torch.rand(
                (batch["metadata"]["num_molid"],),
                device=self.device,
            )[:, None]
            * T
            + 1
        )
        alpha_t, alpha_tm, beta_t = self._get_noise_schedule(t, T)
        batch["features"]["timestep_encoding"] = t / T
        # Noise the atomic coordinates for score matching
        z = torch.randn_like(features["sdf_coordinates"]) * 2
        batch["features"]["noised_atm_coords"] = self._forward_diffuse_alpha_t(
            features["sdf_coordinates"],
            z,
            alpha_t[indexer["gather_idx_i_molid"]],
        )

        scores = self.score_head(batch)
        # [3, N_pairs] -> [N_pairs, 3]
        pred_fapc = self.compute_internal_coord_samples(
            scores["final_coords_atm"], batch["indexer"]
        ).transpose(0, 1)
        ref_fapc = self.compute_internal_coord_samples(
            features["sdf_coordinates"], batch["indexer"]
        ).transpose(0, 1)
        fape_mask = (ref_fapc - pred_fapc).detach().norm(dim=-1) < 10
        lambda_weighting = (alpha_t * beta_t).sqrt() / (1 - alpha_t)
        pred_pair_dist = (
            (
                scores["final_coords_atm"][indexer["gather_idx_uv_u"]]
                - scores["final_coords_atm"][indexer["gather_idx_uv_v"]]
            )
            .square()
            .sum(dim=-1)
            .add(1e-4)
            .sqrt()
        )
        ref_pair_dist = (
            (
                features["sdf_coordinates"][indexer["gather_idx_uv_u"]]
                - features["sdf_coordinates"][indexer["gather_idx_uv_v"]]
            )
            .square()
            .sum(dim=-1)
            .add(1e-4)
            .sqrt()
        )
        drmsd_loss = (
            (pred_pair_dist - ref_pair_dist)
            .square()
            .sum(dim=-1)
            .add(1e-4)
            .sqrt()
            .sub(1e-2)
            .mul(
                lambda_weighting[
                    indexer["gather_idx_i_molid"][indexer["gather_idx_uv_u"]]
                ]
            )
            .mean()
        )
        fape_loss = (
            (pred_fapc - ref_fapc)[fape_mask]
            .square()
            .sum(dim=-1)
            .add(1e-4)
            .sqrt()
            .mean()
        )
        return drmsd_loss, fape_loss

    def step(self, batch, batch_idx, stage, batch_type):
        batch = self.forward(batch)
        if batch_type == "supervised":
            reg_loss, aux_loss = self.compute_supervised_loss(batch)
            loss = reg_loss + 0.1 * aux_loss
            self.log(
                f"{stage}_loss_regression",
                reg_loss.detach(),
                on_epoch=True,
                batch_size=batch["metadata"]["num_molid"],
            )
            self.log(
                f"{stage}_loss_aux_classification",
                aux_loss.detach(),
                on_epoch=True,
                batch_size=batch["metadata"]["num_molid"],
            )
        elif batch_type == "conformer":
            out_dist = self.task_head(batch["features"]["lig_prop_attr"])
            observations = self.compute_internal_coord_samples(
                batch["features"]["sdf_coordinates"][:, None, :], batch["indexer"]
            )
            indexer = batch["indexer"]
            metadata = batch["metadata"]
            pair_nll = -out_dist.log_prob(observations).mean(dim=0)
            marginal_loss = segment_mean(
                pair_nll, indexer["gather_idx_Uijk_ijk"], metadata["num_ijk"]
            ).mean()
            drmsd_loss, fape_loss = self.compute_DDPM_loss(batch)
            loss = marginal_loss + drmsd_loss / 4 + fape_loss / 4
            self.log(
                f"{stage}_loss_marginal",
                marginal_loss.detach(),
                on_epoch=True,
                batch_size=batch["metadata"]["num_molid"],
            )
            self.log(
                f"{stage}_drmsd_loss",
                drmsd_loss.detach(),
                on_epoch=True,
                batch_size=batch["metadata"]["num_molid"],
            )
            self.log(
                f"{stage}_fape_loss",
                fape_loss.detach(),
                on_epoch=True,
                batch_size=batch["metadata"]["num_molid"],
            )
        else:
            raise NotImplementedError
        mlm_loss_atom, mlm_loss_pair = self.compute_mlm_loss(batch)
        self.log(
            f"{stage}_loss_mlm_atom",
            mlm_loss_atom.detach(),
            on_epoch=True,
            batch_size=batch["metadata"]["num_molid"],
        )
        self.log(
            f"{stage}_loss_mlm_pair",
            mlm_loss_pair.detach(),
            on_epoch=True,
            batch_size=batch["metadata"]["num_molid"],
        )
        loss = loss + mlm_loss_atom + mlm_loss_pair
        self.log(
            f"{stage}_loss",
            loss.detach(),
            on_epoch=True,
            batch_size=batch["metadata"]["num_molid"],
        )
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = 0
        for batch_type in ["supervised", "conformer"]:
            loss = loss + self.step(batch[batch_type], batch_idx, "train", batch_type)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = 0
        for batch_type in ["supervised", "conformer"]:
            loss = loss + self.step(batch[batch_type], batch_idx, "val", batch_type)
        if not torch.is_tensor(loss):
            return None
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        self.eval()
        loss = 0
        for batch_type in ["supervised", "conformer"]:
            loss = loss + self.step(batch[batch_type], batch_idx, "test", batch_type)
        if not torch.is_tensor(loss):
            return None
        return {"loss": loss}

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

        # skip the first 1000 steps
        if self.trainer.global_step < 1000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 1000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.task_config.init_learning_rate

    def on_save_checkpoint(self, checkpoint):
        # Omit supervised head params to save disk space
        for key in list(checkpoint["state_dict"].keys()):
            if key.startswith("global_readout"):
                del checkpoint["state_dict"][key]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.task_config.init_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.task_config.max_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "train_loss",
        }


class MolGeomPredictionHead(nn.Module):
    def __init__(
        self,
        dim,
        pair_dim,
        n_stacks=3,
        n_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.pair_dim = pair_dim
        self.atm_embed = GELUMLP(dim + 64, pair_dim)
        self.trp_embed = nn.Linear(dim, pair_dim, bias=False)
        self.trp_vembed = nn.Linear(dim, pair_dim * 3, bias=False)
        self.net = nn.ModuleList(
            [
                EquivariantTransformerBlock(
                    pair_dim,
                    heads=n_heads,
                    point_dim=pair_dim // (n_heads * 2),
                    edge_dim=pair_dim,
                    target_frames=True,
                )
                for _ in range(n_stacks)
            ]
        )
        self.time_encoding = GaussianFourierEncoding1D(32)
        self.out_atm = nn.Linear(pair_dim, 1, bias=False)
        self.out_scale = nn.Linear(pair_dim, 1)

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
    ):
        features = batch["features"]
        indexer = batch["indexer"]
        metadata = batch["metadata"]

        timestep_mol = features["timestep_encoding"][indexer["gather_idx_i_molid"]]
        atm_coords = features["noised_atm_coords"]

        atm_rep_int = self.atm_embed(
            torch.cat(
                [
                    features["lig_atom_attr"],
                    self.time_encoding(timestep_mol),
                ],
                dim=-1,
            )
        )
        atm_rep = self._init_scalar_vec_rep(atm_rep_int)
        triplet_ijk = (
            indexer["gather_idx_ijk_i"][indexer["gather_idx_I_ijk"]],
            indexer["gather_idx_ijk_j"][indexer["gather_idx_I_ijk"]],
            indexer["gather_idx_ijk_k"][indexer["gather_idx_I_ijk"]],
        )
        triplet_frames = get_frame_matrix(
            atm_coords[triplet_ijk[0]],
            atm_coords[triplet_ijk[1]],
            atm_coords[triplet_ijk[2]],
        )
        trp_rep = self._init_scalar_vec_rep(
            self.trp_embed(features["lig_trp_attr"]),
            self.trp_vembed(features["lig_trp_attr"]),
            frame=triplet_frames,
        )
        # at_edge_rep = self.edge_embedding(features["prop_attr"])
        at_edge_rep = features["lig_af_pair_attr"]
        aa_edge_rep = features["lig_atom_pair_attr"]

        node_coords = {
            "atm_t": atm_coords,
        }
        # 09/01: test the batcher on this binary problem
        graph_relations = [
            (
                "triplet_to_atm",
                "gather_idx_UI_I",
                "gather_idx_UI_u",
                "trp",
                "atm",
            ),
            (
                "atm_to_triplet",
                "gather_idx_UI_u",
                "gather_idx_UI_I",
                "atm",
                "trp",
            ),
            (
                "atm_to_atm",
                "gather_idx_uv_u",
                "gather_idx_uv_v",
                "atm",
                "atm",
            ),
        ]
        # Single-type node counting must be handled externally
        metadata["num_atm"] = metadata["num_i"]
        metadata["num_trp"] = metadata["num_I"]
        graph_batcher = make_multi_relation_graph_batcher(
            graph_relations, indexer, metadata
        )
        merged_atm_trp_rep = graph_batcher.collate_node_attr(
            {"atm": atm_rep, "trp": trp_rep}
        )
        merged_edge_idx = graph_batcher.collate_idx_list(indexer)

        for subnet in self.net:
            # iterative update
            triplet_frames = get_frame_matrix(
                node_coords["atm_t"][triplet_ijk[0]],
                node_coords["atm_t"][triplet_ijk[1]],
                node_coords["atm_t"][triplet_ijk[2]],
            )
            dummy_atm_frames = RigidTransform(node_coords["atm_t"], R=None)
            merged_atm_trp_rep, _ = subnet(
                merged_atm_trp_rep,
                merged_edge_idx,
                graph_batcher.collate_node_attr(
                    {"atm": dummy_atm_frames.t, "trp": triplet_frames.t}
                ),
                R=graph_batcher.collate_node_attr(
                    {"atm": dummy_atm_frames.R, "trp": triplet_frames.R}
                ),
                x_edge=graph_batcher.collate_edge_attr(
                    {
                        "triplet_to_atm": at_edge_rep,
                        "atm_to_triplet": at_edge_rep,
                        "atm_to_atm": aa_edge_rep,
                    }
                ),
            )
            atm_rep = graph_batcher.offload_node_attr(merged_atm_trp_rep)["atm"]
            # Displacement vectors in the global coordinate system
            drift_scale = torch.sigmoid(self.out_scale(atm_rep[:, 0])) * 10
            drift_dir = self.out_atm(atm_rep[:, 1:]).squeeze(-1)
            drift_atm = drift_dir * drift_scale
            node_coords["atm_t"] = node_coords["atm_t"] + drift_atm
        return {
            "final_coords_atm": node_coords["atm_t"],
        }
