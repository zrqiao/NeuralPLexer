import math
from typing import Tuple

import torch
from torch import Tensor, nn

from neuralplexer.util.frame import RigidTransform


def smoothstep(xin, edge0, edge1):
    x = (xin - edge0) / (edge1 - edge0)
    out = torch.zeros_like(x)
    out[(x > 0) & (x < 1)] = (
        3 * x[(x > 0) & (x < 1)] ** 2 - 2 * x[(x > 0) & (x < 1)] ** 3
    )
    out[x > 1] = 1
    return out


def bessel_dist_encoding(r, f, r0):
    return torch.sin(math.pi * f * r / r0) / (r + 1)


def gaussian_dist_encoding(r, f, r0):
    bins = f / f.max() * r0
    sigma = r0 / f.max()
    return torch.exp(-(((bins - r) / sigma) ** 2))


def fourier_dist_encoding(r, f, r0):
    return torch.sin(math.pi * f * r / r0)


class BinnedOnehotEncoding1D(nn.Module):
    def __init__(
        self,
        bin_vmax,
    ):
        super().__init__()
        self.bins = nn.Parameter(torch.tensor(bin_vmax), requires_grad=False)

    def forward(
        self,
        x: Tensor,
    ):
        encodings = (x.unsqueeze(-1) < self.bins).float()
        return encodings


class GaussianRBFEncoding1D(nn.Module):
    def __init__(
        self,
        n_basis,
        x_max,
        sigma=1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.rbf_centers = nn.Parameter(
            torch.linspace(0, x_max, n_basis), requires_grad=False
        )

    def forward(
        self,
        x: Tensor,
    ):
        encodings = torch.exp(
            -((x.unsqueeze(-1) - self.rbf_centers).div(self.sigma).square())
        )
        return encodings


class GaussianFourierEncoding1D(nn.Module):
    def __init__(
        self,
        n_basis,
        eps=1e-2,
    ):
        super().__init__()
        self.eps = eps
        self.fourier_freqs = nn.Parameter(torch.randn(n_basis) * math.pi)

    def forward(
        self,
        x: Tensor,
    ):
        encodings = torch.cat(
            [
                torch.sin(self.fourier_freqs.mul(x)),
                torch.cos(self.fourier_freqs.mul(x)),
            ],
            dim=-1,
        )
        return encodings


class GaussianBesselEncoding3D(nn.Module):
    def __init__(
        self,
        n_basis,
        dim,
        eps=1.0,
    ):
        super().__init__()
        self.eps = eps
        self.fourier_freqs = nn.Parameter(torch.randn(n_basis) * math.pi)
        self.out = nn.Linear(n_basis * 6, dim, bias=False)

    def forward(
        self,
        vecs: Tensor,
    ):
        dists = vecs.square().sum(-1, keepdim=True).add(self.eps).sqrt()
        dirs = vecs / dists
        dist_encodings = torch.cat(
            [
                torch.sin(self.fourier_freqs.mul(dists)),
                torch.cos(self.fourier_freqs.mul(dists)),
            ],
            dim=-1,
        )
        vec_encodings = (
            dist_encodings.unsqueeze(-1).mul(dirs.unsqueeze(-2)).flatten(-2, -1) / dists
        )
        return self.out(vec_encodings)


class RelativeGeometryEncoding(nn.Module):
    "Compute radial basis functions and iterresidue/pseudoresidue orientations."

    def __init__(self, n_basis, out_dim, d_max=20.0):
        super(RelativeGeometryEncoding, self).__init__()
        self.rbf_encoding = GaussianRBFEncoding1D(n_basis, d_max)
        self.rel_geom_projector = nn.Linear(n_basis + 15, out_dim, bias=False)

    def forward(self, frames: RigidTransform, merged_edge_idx: Tuple[Tensor, Tensor]):
        frame_t, frame_R = frames.t, frames.R
        pair_dists = torch.norm(
            frame_t[merged_edge_idx[0]] - frame_t[merged_edge_idx[1]],
            dim=-1,
        )
        pair_directions_l = torch.matmul(
            (frame_t[merged_edge_idx[1]] - frame_t[merged_edge_idx[0]]).unsqueeze(-2),
            frame_R[merged_edge_idx[0]],
        ).squeeze(-2) / pair_dists.square().add(1).sqrt().unsqueeze(-1)
        pair_directions_r = torch.matmul(
            (frame_t[merged_edge_idx[0]] - frame_t[merged_edge_idx[1]]).unsqueeze(-2),
            frame_R[merged_edge_idx[1]],
        ).squeeze(-2) / pair_dists.square().add(1).sqrt().unsqueeze(-1)
        pair_orientations = torch.matmul(
            frame_R.transpose(-2, -1).contiguous()[merged_edge_idx[0]],
            frame_R[merged_edge_idx[1]],
        )
        return self.rel_geom_projector(
            torch.cat(
                [
                    self.rbf_encoding(pair_dists),
                    pair_directions_l,
                    pair_directions_r,
                    pair_orientations.flatten(-2, -1),
                ],
                dim=-1,
            )
        )
