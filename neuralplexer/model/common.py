import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_max, scatter_min


class GELUMLP(nn.Module):
    """Simple MLP with post-LayerNorm"""

    def __init__(
        self,
        n_in_feats,
        n_out_feats,
        n_hidden_feats=None,
        dropout=0.0,
        zero_init=False,
    ):
        super(GELUMLP, self).__init__()
        self.dropout = dropout
        if n_hidden_feats is None:
            self.layers = nn.Sequential(
                nn.Linear(n_in_feats, n_in_feats),
                nn.GELU(),
                nn.LayerNorm(n_in_feats),
                nn.Linear(n_in_feats, n_out_feats),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_in_feats, n_hidden_feats),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(n_hidden_feats, n_hidden_feats),
                nn.GELU(),
                nn.LayerNorm(n_hidden_feats),
                nn.Linear(n_hidden_feats, n_out_feats),
            )
        nn.init.xavier_uniform_(self.layers[0].weight, gain=1)
        # zero init for residual branches
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
        else:
            nn.init.xavier_uniform_(self.layers[-1].weight, gain=1)

    def _zero_init(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)


def segment_logsumexp(src, dst_idx, dst_size, extra_dims=None):
    src_max, _ = scatter_max(src, dst_idx, dim=0, dim_size=dst_size)
    if extra_dims is not None:
        src_max = torch.amax(src_max, dim=extra_dims, keepdim=True)
    src = src - src_max[dst_idx]
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, torch.exp(src))
    if extra_dims is not None:
        out = torch.sum(out, dim=extra_dims)
    return torch.log(out + 1e-8) + src_max.view(*out.shape)


def segment_softmax(src, dst_idx, dst_size, extra_dims=None, floor_value=None):
    src_max, _ = scatter_max(src, dst_idx, dim=0, dim_size=dst_size)
    if extra_dims is not None:
        src_max = torch.amax(src_max, dim=extra_dims, keepdim=True)
    src = src - src_max[dst_idx]
    exp1 = torch.exp(src)
    exp0 = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, exp1)
    if extra_dims is not None:
        exp0 = torch.sum(exp0, dim=extra_dims, keepdim=True)
    exp0 = torch.index_select(input=exp0, dim=0, index=dst_idx)
    exp = exp1.div(exp0 + 1e-8)
    if floor_value is not None:
        exp = exp.clamp(min=floor_value)
        exp0 = torch.zeros(
            dst_size,
            *src.shape[1:],
            dtype=src.dtype,
            device=src.device,
        ).index_add_(0, dst_idx, exp)
        if extra_dims is not None:
            exp0 = torch.sum(exp0, dim=extra_dims, keepdim=True)
        exp0 = torch.index_select(input=exp0, dim=0, index=dst_idx)
        exp = exp.div(exp0 + 1e-8)
    return exp


def batched_sample_onehot(logits, dim=0, max_only=False):
    # The Gumbel-max trick
    if max_only:
        sampled_idx = torch.argmax(logits, dim=dim, keepdim=True)
    else:
        noise = torch.rand_like(logits)
        sampled_idx = torch.argmax(
            logits - torch.log(-torch.log(noise)), dim=dim, keepdim=True
        )
    out_onehot = torch.zeros_like(logits, dtype=torch.bool)
    out_onehot.scatter_(dim=dim, index=sampled_idx, value=1)
    return out_onehot


def segment_sample_onehot(logits, dst_idx, dst_size, max_only=False):
    # The Gumbel-max trick
    if max_only:
        _, sampled_idx = scatter_max(logits, dst_idx, dim=0, dim_size=dst_size)
    else:
        noise = torch.rand_like(logits)
        _, sampled_idx = scatter_max(
            logits - torch.log(-torch.log(noise)), dst_idx, dim=0, dim_size=dst_size
        )
    out_onehot = torch.zeros_like(logits, dtype=torch.bool)
    out_onehot[sampled_idx] = 1
    return out_onehot


def segment_argmax(scores, dst_idx, dst_size, randomize=False):
    if randomize:
        noise = torch.rand_like(scores)
        scores = scores - torch.log(-torch.log(noise))
    _, sampled_idx = scatter_max(scores, dst_idx, dim=0, dim_size=dst_size)
    return sampled_idx


def segment_argmin(scores, dst_idx, dst_size, randomize=False):
    if randomize:
        noise = torch.rand_like(scores)
        scores = scores - torch.log(-torch.log(noise))
    _, sampled_idx = scatter_min(scores, dst_idx, dim=0, dim_size=dst_size)
    return sampled_idx


def segment_topk(scores, k, dst_idx, dst_size, src_sizes, randomize=False):
    # Padding-based segment-wise top-k pooling
    if randomize:
        noise = torch.rand_like(scores)
        scores = scores - torch.log(-torch.log(noise))
    segment_offsets = torch.cumsum(src_sizes, dim=0) - src_sizes
    src_idx_in_segment = (
        torch.arange(scores.shape[0], device=scores.device) - segment_offsets
    )
    scatter_idx = dst_idx * max(src_sizes) + src_idx_in_segment
    padded_scores = torch.full(dst_size * max(src_sizes), -1e8, device=scores.device)
    padded_scores[scatter_idx] = scores
    padded_scores = padded_scores.view(dst_size, max(src_sizes))
    topk_values, topk_idx = torch.topk(padded_scores, k, dim=1, largest=True)
    # Note that returned here is the indices in segment, offset subtracted
    topk_mask = topk_values < -5e7
    return topk_idx, topk_mask


def topk_edge_mask_from_logits(scores, k, randomize=False):
    # [B, N, N]
    assert len(scores.shape) == 3
    if randomize:
        noise = torch.rand_like(scores)
        scores = scores - torch.log(-torch.log(noise))
    node_degree = min(k, scores.shape[2])
    topk_values, topk_idx = torch.topk(scores, node_degree, dim=-1, largest=True)
    edge_mask = scores.new_zeros(scores.shape, dtype=torch.bool)
    edge_mask = edge_mask.scatter_(dim=2, index=topk_idx, value=1).bool()
    return edge_mask


def masked_softmax(src, mask, dim):
    src_max = torch.amax(src, dim=dim, keepdim=True)
    exps = torch.zeros_like(src)
    exps[~mask] = torch.exp((src - src_max)[~mask])
    masked_sums = exps.sum(dim, keepdim=True) + 1e-12
    return exps / masked_sums


def masked_softplus(src, mask, dim):
    sps = torch.zeros_like(src)
    sps[~mask] = F.softplus(src[~mask])
    return sps


def segment_sum(src, dst_idx, dst_size):
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, src)
    return out


def segment_min(src, dst_idx, dst_size):
    out, _ = scatter_min(src, dst_idx, dim=0, dim_size=dst_size)
    return out


def segment_mean(src, dst_idx, dst_size):
    out = torch.zeros(
        dst_size,
        *src.shape[1:],
        dtype=src.dtype,
        device=src.device,
    ).index_add_(0, dst_idx, src)
    denom = (
        torch.zeros(
            dst_size,
            *src.shape[1:],
            dtype=src.dtype,
            device=src.device,
        ).index_add_(0, dst_idx, torch.ones_like(src))
        + 1e-8
    )
    return out / denom


class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dst_idx, dst_size):
        """"""

        return segment_sum(x, dst_idx, dst_size)


class AveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dst_idx, dst_size):
        """"""

        out = torch.zeros(
            dst_size,
            *x.shape[1:],
            dtype=x.dtype,
            device=x.device,
        ).index_add_(0, dst_idx, x)
        nmr = torch.zeros(
            dst_size,
            *x.shape[1:],
            dtype=x.dtype,
            device=x.device,
        ).index_add_(0, dst_idx, torch.ones_like(x))

        return out / (nmr + 1e-8)
