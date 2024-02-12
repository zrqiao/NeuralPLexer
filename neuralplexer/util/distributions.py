"""Density estimation modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import PowerSpherical
from torch.distributions import Normal, OneHotCategorical

from neuralplexer.model.common import GELUMLP


class MixtureDensityNetwork3D(nn.Module):
    """
    Mixture density network. [ Bishop, 1994 ]
    Adapted from https://github.com/tonyduan/mdn/blob/master/mdn/models.py

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    n_modes: int; number of components in the mixture model
    """

    def __init__(self, dim_in, n_modes):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_modes)
        self.normal_network = MixtureNormalNetwork3D(dim_in, n_modes)

    def forward(self, x, normalize_logits=True):
        return (
            self.pi_network(x, normalize_logits=normalize_logits),
            self.normal_network(x),
        )

    @staticmethod
    def normal_invscale_log_prob(mean, logu_diag, sd_tril, y):
        # See: https://arxiv.org/abs/2003.05739
        diff = y - mean
        logjac = logu_diag.sum(-1)
        mah_dist = diff.unsqueeze(-1).mul(sd_tril).sum(-2)
        # mah_dist = sd_tril.mul(diff.unsqueeze(-2)).sum(-1)
        return (-1 / 2) * mah_dist.pow(2).sum(-1) + logjac - 2.75681559961

    def neg_log_likelihood(self, pi_params, normal_params, y):
        # The loc shape is [n_samples, n_props, n_modes, 3]
        pi_params = pi_params.unsqueeze(0)
        mean, logu_diag, sd_tril = normal_params
        loglik = self.normal_invscale_log_prob(
            mean.unsqueeze(0),
            logu_diag.unsqueeze(0),
            sd_tril.unsqueeze(0),
            y.unsqueeze(2),
        )
        nll = -torch.logsumexp(pi_params + loglik, dim=2)
        return nll

    def sample(self, pi_params, normal_params, n):
        # Assume [n_pairs, n_modes, 3]
        pi_dist = torch.distributions.Categorical(logits=pi_params)
        mode_samples = pi_dist.sample(sample_shape=(n,)).T
        mean, logu_diag, sd_tril = normal_params
        a = sd_tril[:, :, 0, 0]
        b = sd_tril[:, :, 1, 0]
        c = sd_tril[:, :, 2, 0]
        d = sd_tril[:, :, 1, 1]
        e = sd_tril[:, :, 2, 1]
        f = sd_tril[:, :, 2, 2]
        # Simply calling the analytical formula
        # https://math.stackexchange.com/questions/1003801/inverse-of-an-invertible-upper-triangular-matrix-of-order-3
        cov_tril = torch.stack(
            [
                torch.stack([1 / a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1),
                torch.stack([-b / (a * d), 1 / d, torch.zeros_like(a)], dim=-1),
                torch.stack(
                    [(b * e - c * d) / (a * d * f), -e / (d * f), 1 / f], dim=-1
                ),
            ],
            dim=-2,
        )
        mean_gathered = torch.gather(
            mean, 1, mode_samples.unsqueeze(-1).expand(-1, -1, 3)
        )
        cov_tril_gathered = torch.gather(
            cov_tril, 1, mode_samples.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        )
        sampled_z = torch.randn_like(mean_gathered)
        sampled_x = (
            sampled_z.unsqueeze(-1).mul(cov_tril_gathered).sum(-2) + mean_gathered
        )
        return sampled_x  # Output shape is [n_pairs, n_samples, 3]


class MixtureNormalNetwork3D(nn.Module):
    def __init__(self, in_dim, n_modes, hidden_dim=None):
        super().__init__()
        self.n_modes = n_modes
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = GELUMLP(in_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 9 * n_modes)

    def forward(self, x):
        params = self.out_layer(self.network(x))
        params = params.view(-1, self.n_modes, 9)
        mean, fu_diag, u_offdiag = torch.split(params, 3, dim=2)
        # u_offdiag = u_offdiag.unsqueeze(-1)
        # mean = torch.sinh(mean)
        # u_diag = F.elu(fu_diag) + 1
        # logu_diag = torch.zeros_like(fu_diag)
        # logu_diag[fu_diag < 0] = fu_diag[fu_diag < 0]
        # logu_diag[fu_diag > 0] = torch.log(fu_diag[fu_diag > 0] + 1)
        u_diag = torch.exp(fu_diag)
        logu_diag = fu_diag
        sd_tril = torch.diag_embed(u_diag)
        # sd_tril = torch.diag_embed(F.softplus(u_diag))
        sd_tril[:, :, (1, 2, 2), (0, 0, 1)] = u_offdiag
        return mean, logu_diag, sd_tril


class GaussianGrid3D(nn.Module):
    def __init__(self, in_dim, rad_res, ang_res, hidden_dim=None):
        super().__init__()
        self.n_modes = 2 * rad_res * (ang_res**2)
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = GELUMLP(in_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, self.n_modes)

    def forward(self, x):
        params = self.out_layer(self.network(x))
        return OneHotCategorical(logits=params)

    def neg_log_likelihood(self, pi, y):
        # The grid index shape is [n_props, n_confs]
        grid_index = self._compute_bin_index(y)
        loglik = torch.gather(pi.logits, 1, grid_index)
        nll = -torch.logsumexp(loglik, dim=1)
        return nll


class CategoricalNetwork(nn.Module):
    def __init__(self, in_dim, n_modes, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.network = GELUMLP(in_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, n_modes)

    def forward(self, x, normalize_logits=True):
        params = self.out_layer(self.network(x))
        if normalize_logits:
            params = params - torch.logsumexp(params, dim=-1, keepdim=True)
        return params


class Morse3Transform:
    def __init__(self, scale):
        self.scale = scale
        self.eps = 1e-6

    def transform(self, r):
        return (1 - torch.exp(-r / self.scale)).pow(3)

    def inverse(self, z):
        return -torch.log1p(-z.pow(1 / 3) + self.eps).mul(self.scale)

    def log_jacobian(self, r):
        return (
            math.log(3 / self.scale)
            + 2 * torch.log1p(-torch.exp(-r / self.scale) + self.eps)
            - r / self.scale
        )


class MorseTransform:
    def __init__(self, scale):
        self.scale = scale
        self.eps = 1e-6

    def transform(self, r):
        return (1 - torch.exp(-r / self.scale)).square()

    def inverse(self, z):
        return -torch.log1p(-z.sqrt() + self.eps).mul(self.scale)

    def log_jacobian(self, r):
        return (
            math.log(2 / self.scale)
            + torch.log1p(-torch.exp(-r / self.scale) + self.eps)
            - r / self.scale
        )


class RadialGaussian1D(torch.distributions.Distribution):
    def __init__(
        self,
        base_dist: torch.distributions.Normal,
        radial_transform=None,
    ):
        super(RadialGaussian1D, self).__init__(validate_args=False)
        self.base_dist = base_dist
        self.radial_transform = radial_transform

    def rsample(self, sample_shape=torch.Size()):
        sampled_z = self.base_dist.rsample(sample_shape)
        sampled_z = torch.clamp(sampled_z, min=1e-3, max=1 - 1e-3)
        if self.radial_transform is None:
            sampled_r = sampled_z
        else:
            sampled_r = self.radial_transform.inverse(sampled_z)
        return sampled_r

    def log_prob(self, r: torch.Tensor):
        if self.radial_transform is None:
            z = r
            radial_transform_log_jacobian = 0
        else:
            # Note that radial_transform is the inverse flow r->z
            z = self.radial_transform.transform(r)
            radial_transform_log_jacobian = self.radial_transform.log_jacobian(r)
        spherical_conversion_log_jacobian = torch.log(r) * 2
        return (
            self.base_dist.log_prob(z)
            - spherical_conversion_log_jacobian
            + radial_transform_log_jacobian
        )

    def bare_log_prob(self, r: torch.Tensor):
        if self.radial_transform is None:
            z = r
        else:
            z = self.radial_transform.transform(r)
        return self.base_dist.log_prob(z)


class JointGPS3D(torch.distributions.Distribution):
    """
    Product of Gaussian and Power Spherical distributions
    for modelling 3D radial-angular distributional data.
    """

    def __init__(
        self,
        radial_dist: torch.distributions.Distribution,
        angular_dist: torch.distributions.Distribution,
    ):
        super(JointGPS3D, self).__init__(validate_args=False)
        self.radial_dist, self.angular_dist = radial_dist, angular_dist
        self._eps = 1e-6

    def rsample(self, sample_shape=torch.Size()):
        sampled_r = self.radial_dist.rsample(sample_shape)
        sampled_uvec = self.angular_dist.rsample(sample_shape)
        return sampled_r.unsqueeze(-1).mul(sampled_uvec)

    def log_prob(self, value: torch.Tensor):
        r = value.square().sum(-1).add(self._eps).sqrt()
        v = value / r.unsqueeze(-1)
        return self.radial_dist.log_prob(r) + self.angular_dist.log_prob(v)

    def entropy(self):
        return self.radial_dist.entropy() + self.angular_dist.entropy()


class MixtureRadialNetwork(nn.Module):
    """
    Mixture density network for radial data.

    Parameters
    ----------
    dim_in: int; dimensionality of the input features
    n_points: int; number of models
    n_modes: int; number of components in the mixture model
    """

    def __init__(self, dim_in, n_points, n_modes, r_0=5):
        super().__init__()
        self.n_points = n_points
        self.n_modes = n_modes
        self.network = GELUMLP(dim_in, dim_in)
        self.pi_layer = nn.Linear(dim_in, n_modes * n_points)
        self.bin_centers = nn.Parameter(
            torch.linspace(1.0 / n_modes, 1, n_modes, dtype=torch.float)
            .expand(n_points, n_modes)
            .clone(),
            requires_grad=False,
        )
        self.mean_layer = nn.Linear(dim_in, n_modes * n_points)
        self.std_layer = nn.Linear(dim_in, n_modes * n_points)
        self.radial_transform = MorseTransform(scale=r_0)
        self._eps = 1e-6

    def forward(self, x: torch.Tensor):
        x = self.network(x)
        pi_params = self.pi_layer(x).unflatten(-1, (self.n_points, self.n_modes))
        radial_bins = self.bin_centers + self.mean_layer(x).unflatten(
            -1, (self.n_points, self.n_modes)
        )
        radial_stds = F.softplus(
            self.std_layer(x).unflatten(-1, (self.n_points, self.n_modes))
        ).add(self._eps)
        radial_dists = RadialGaussian1D(
            Normal(radial_bins, radial_stds),
            radial_transform=self.radial_transform,
        )
        return MixtureGPSDistribution(pi_params, radial_dists)


class MixtureGPSNetwork3D(nn.Module):
    """
    Mixture density network over gaussian - power spherical distributions.

    Parameters
    ----------
    dim_in: int; dimensionality of the input features
    n_modes: int; number of components in the mixture model
    """

    def __init__(self, dim_in, n_modes, r_0=16):
        super().__init__()
        self.n_modes = n_modes
        self.network = GELUMLP(dim_in, dim_in)
        self.pi_layer = nn.Linear(dim_in, n_modes)
        # self.bin_centers = nn.Parameter(
        #     torch.linspace(0, r_0, n_modes, dtype=torch.float),
        #     requires_grad=True,
        # )
        self.bin_centers = nn.Parameter(
            torch.linspace(0, 1, n_modes, dtype=torch.float),
            requires_grad=False,
        )
        self.mean_layer = nn.Linear(dim_in, n_modes)
        self.std_layer = nn.Linear(dim_in, n_modes)
        self.mu_layer = nn.Linear(dim_in, 3 * n_modes)
        self.radial_transform = MorseTransform(scale=r_0)
        self._eps = 1e-6

    def forward(self, x: torch.Tensor):
        x = self.network(x)
        pi_params = self.pi_layer(x)
        mu_orig = self.mu_layer(x).unflatten(-1, (self.n_modes, 3))
        scale = mu_orig.square().sum(dim=-1)
        mu_norm = scale.add(self._eps).sqrt().unsqueeze(-1)
        mu = mu_orig / mu_norm
        radial_bins = self.bin_centers + self.mean_layer(x)
        radial_stds = F.softplus(self.std_layer(x)).add(self._eps)
        radial_dists = RadialGaussian1D(
            Normal(radial_bins, radial_stds),
            radial_transform=self.radial_transform,
        )
        angular_dists = PowerSpherical(mu, scale)
        comp_dists = JointGPS3D(radial_dists, angular_dists)
        return MixtureGPSDistribution(pi_params, comp_dists)


class JointGPS6D(torch.distributions.Distribution):
    """
    Product of Gaussian and Power Spherical distributions
    for modelling 3D radial-angular distributional data.
    """

    def __init__(
        self,
        radial_dist: torch.distributions.Distribution,
        angular_dist: torch.distributions.Distribution,
        orientation_dist: torch.distributions.Distribution,
    ):
        super(JointGPS6D, self).__init__(validate_args=False)
        self.pos_dist = JointGPS3D(radial_dist, angular_dist)
        self.orientation_dist = orientation_dist

    def rsample(self, sample_shape=torch.Size()):
        return torch.cat(
            (
                self.pos_dist.rsample(sample_shape),
                self.orientation_dist.rsample(sample_shape),
            ),
            dim=-1,
        )

    def log_prob(self, value: torch.Tensor):
        # x, y, z, q_r, q_i, q_j, q_k
        assert value.shape[-1] == 7
        pos = value[..., :3]
        ori = value[..., 3:]
        ori_logprob = torch.maximum(
            self.orientation_dist.log_prob(ori),
            self.orientation_dist.log_prob(-ori),
        )
        return self.pos_dist.log_prob(pos) + ori_logprob

    def log_prob_pos(self, value: torch.Tensor):
        assert value.shape[-1] == 7
        pos = value[..., :3]
        return self.pos_dist.log_prob(pos)

    def entropy(self):
        return self.pos_dist.entropy() + self.orientation_dist.entropy()


class MixtureGPSNetwork6D(nn.Module):
    """
    Mixture density network for position + orientation modelling

    Parameters
    ----------
    dim_in: int; dimensionality of the input features
    n_modes: int; number of components in the mixture model
    """

    def __init__(self, dim_in, n_modes, r_0=16):
        super().__init__()
        self.n_modes = n_modes
        self.network = GELUMLP(dim_in, dim_in)
        self.pi_layer = nn.Linear(dim_in, n_modes)
        # self.bin_centers = nn.Parameter(
        #     torch.linspace(0, r_0, n_modes, dtype=torch.float),
        #     requires_grad=True,
        # )
        self.bin_centers = nn.Parameter(
            torch.linspace(0, 1, n_modes, dtype=torch.float),
            requires_grad=False,
        )
        self.mean_layer = nn.Linear(dim_in, n_modes)
        self.std_layer = nn.Linear(dim_in, n_modes)
        self.mu_layer = nn.Linear(dim_in, 3 * n_modes)
        self.muq_layer = nn.Linear(dim_in, 4 * n_modes)
        self.radial_transform = MorseTransform(scale=r_0)
        self._eps = 1e-6

    def forward(self, x: torch.Tensor, raw=False, **kwargs):
        x = self.network(x)
        pi_params = self.pi_layer(x)
        radial_bins = self.bin_centers + self.mean_layer(x)
        radial_stds = F.softplus(self.std_layer(x)).add(0.01) / self.n_modes
        mu_orig = self.mu_layer(x).unflatten(-1, (self.n_modes, 3))
        muq_orig = self.muq_layer(x).unflatten(-1, (self.n_modes, 4))
        if raw:
            return torch.cat(
                [
                    pi_params.unsqueeze(-1),
                    radial_bins.unsqueeze(-1),
                    radial_stds.unsqueeze(-1),
                    mu_orig,
                    muq_orig,
                ],
                dim=-1,
            )
        comp_dists = self.get_comp_distributions(
            radial_bins, radial_stds, mu_orig, muq_orig
        )
        return MixtureGPSDistribution(pi_params, comp_dists, **kwargs)

    def get_comp_distributions(self, radial_bins, radial_stds, mu_orig, muq_orig):
        radial_dists = RadialGaussian1D(
            Normal(radial_bins, radial_stds),
            radial_transform=self.radial_transform,
        )
        scale = mu_orig.square().sum(dim=-1)
        mu_norm = scale.add(self._eps).sqrt().unsqueeze(-1)
        mu = mu_orig / mu_norm
        angular_dists = PowerSpherical(mu, scale)
        scale_q = muq_orig.square().sum(dim=-1)
        muq_norm = scale_q.add(self._eps).sqrt().unsqueeze(-1)
        muq = muq_orig / muq_norm
        orient_dists = PowerSpherical(muq, scale_q)
        comp_dists = JointGPS6D(
            radial_dists,
            angular_dists,
            orient_dists,
        )
        return comp_dists


class MixtureGPSDistribution:
    """Tentative implementation"""

    def __init__(self, pi_params, comp_dists, normalize_logits=True, **kwargs):
        if normalize_logits:
            pi_params = pi_params - torch.logsumexp(pi_params, dim=-1, keepdim=True)
        self.pi_params = pi_params
        self.comp_dists = comp_dists

    def log_prob(self, y, n_d=1):
        # [sample_shape, ..., n_modes, d]
        comp_lls = self.comp_dists.log_prob(y.unsqueeze(-1 - n_d))
        return torch.logsumexp(self.pi_params + comp_lls, dim=-1)

    def bare_log_prob(self, y, n_d=1):
        comp_bare_lls = self.comp_dists.bare_log_prob(y.unsqueeze(-1 - n_d))
        return torch.logsumexp(self.pi_params + comp_bare_lls, dim=-1)

    def sample(self, sample_shape: torch.Size):
        # Memory-inefficient implementation
        pi_dist = torch.distributions.Categorical(logits=self.pi_params)
        mode_samples = pi_dist.sample(sample_shape=sample_shape)
        comp_samples = self.comp_dists.rsample(sample_shape)
        comp_samples_gathered = torch.gather(
            comp_samples, -2, mode_samples.unsqueeze(-1).expand(*comp_samples.shape)
        )
        return comp_samples_gathered
