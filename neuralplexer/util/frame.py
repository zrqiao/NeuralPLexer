"""Atom-mapping and coordinate transformation related operations"""
import torch
from pytorch3d.transforms import matrix_to_quaternion


class RigidTransform:
    def __init__(self, t: torch.Tensor, R: torch.Tensor = None):
        self.t = t
        if R is None:
            R = t.new_zeros(*t.shape, 3)
        self.R = R

    def __getitem__(self, key):
        return RigidTransform(self.t[key], self.R[key])

    def unsqueeze(self, dim):
        return RigidTransform(self.t.unsqueeze(dim), self.R.unsqueeze(dim))

    def squeeze(self, dim):
        return RigidTransform(self.t.squeeze(dim), self.R.squeeze(dim))

    def concatenate(self, other, dim=0):
        return RigidTransform(
            torch.cat([self.t, other.t], dim=dim),
            torch.cat([self.R, other.R], dim=dim),
        )


def get_frame_matrix(
    ri: torch.Tensor, rj: torch.Tensor, rk: torch.Tensor, eps=1e-4, strict=False
):
    # Regularized Gram-Schmidt
    # Allow for shearing
    v1 = ri - rj
    v2 = rk - rj
    if strict:
        # v1 = v1 + torch.randn_like(rj).mul(eps)
        # v2 = v2 + torch.randn_like(rj).mul(eps)
        e1 = v1 / v1.norm(dim=-1, keepdim=True)
        # Project and pad
        u2 = v2 - e1.mul(e1.mul(v2).sum(-1, keepdim=True))
        e2 = u2 / u2.norm(dim=-1, keepdim=True)
    else:
        e1 = v1 / v1.square().sum(dim=-1, keepdim=True).add(eps).sqrt()
        # Project and pad
        u2 = v2 - e1.mul(e1.mul(v2).sum(-1, keepdim=True))
        e2 = u2 / u2.square().sum(dim=-1, keepdim=True).add(eps).sqrt()
    e3 = torch.cross(e1, e2, dim=-1)
    # Rows - lab frame, columns - internal frame
    rot_j = torch.stack([e1, e2, e3], dim=-1)
    return RigidTransform(rj, torch.nan_to_num(rot_j, 0.0))


def relative_orientation(o2: torch.Tensor, o1: torch.Tensor):
    # O1^{-1} @ O2
    return torch.matmul(o1.transpose(-2, -1), o2)


def cartesian_to_internal(rs: torch.Tensor, frames: RigidTransform):
    # Right-multiply the pose matrix
    rs_loc = rs - frames.t
    rs_loc = torch.matmul(rs_loc.unsqueeze(-2), frames.R)
    return rs_loc.squeeze(-2)


def internal_to_cartesian(rs_loc: torch.Tensor, frames: RigidTransform):
    # Left-multiply the pose matrix
    rs = torch.matmul(rs_loc.unsqueeze(-2), frames.R.transpose(-2, -1))
    rs = rs.squeeze(-2) + frames.t
    return rs


def compact_se3_transform(tq, TR):
    xyz, R = TR.t, TR.R
    assert tq.shape[-1] == 7
    t, q = tq[..., :3], tq[..., 3:]
    t_lab = torch.matmul(R, t.unsqueeze(-1))
    Q = quaternion_to_rotmat(q)
    # right multiply Q (body-fixed rotation)
    xyz_new = xyz + t_lab.squeeze(-1)
    R_new = torch.matmul(R, Q)
    return RigidTransform(xyz_new, R_new)


def compose_se3_transforms(t1, R1, t2, R2):
    # left multiply R1 (lab-fixed rotation)
    t_out = t1 + torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1)
    R_out = torch.matmul(R1, R2)
    return t_out, R_out


def inv_compose_se3_transforms(t1, R1, t2, R2):
    # left multiply R1 (lab-fixed rotation)
    t_out = torch.matmul(R1.transpose(-2, -1), (t2 - t1).unsqueeze(-1)).squeeze(-1)
    R_out = torch.matmul(R1.transpose(-2, -1), R2)
    return t_out, R_out


def relative_point_set_transform(xyz, source_frame, target_frame, tq):
    assert tq.shape[-1] == 7
    t, q = tq[..., :3], tq[..., 3:]
    source_xyz = cartesian_to_internal(xyz, source_frame)
    target_transform = compact_se3_transform(tq, target_frame[0], target_frame[1])
    xyz_new = internal_to_cartesian(source_xyz, target_transform)
    return xyz_new


def quaternion_to_rotmat(q, eps=1e-6):
    # rescale to unit quaternions
    q = q + eps
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    qr, qi, qj, qk = (
        q[..., 0],
        q[..., 1],
        q[..., 2],
        q[..., 3],
    )
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (qj**2 + qk**2),
                    2 * (qi * qj - qk * qr),
                    2 * (qi * qk + qj * qr),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qi * qj + qk * qr),
                    1 - 2 * (qi**2 + qk**2),
                    2 * (qj * qk - qi * qr),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qi * qk - qj * qr),
                    2 * (qj * qk + qi * qr),
                    1 - 2 * (qi**2 + qj**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def rotmat_to_quaternion(R: torch.Tensor):
    return matrix_to_quaternion(R)


def apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X


if __name__ == "__main__":
    for _ in range(100):
        q = torch.randn(4)
        q = q / torch.norm(q)
        R = quaternion_to_rotmat(q)
        q1 = rotmat_to_quaternion(R)
        try:
            assert torch.all(torch.isclose(q, q1, atol=1e-5))
        except AssertionError:
            assert torch.all(torch.isclose(q, -q1, atol=1e-5))
