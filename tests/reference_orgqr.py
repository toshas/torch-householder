import torch


def reference_orgqr(v, tau, checks=True):
    """
    Unoptimized differentiable implementation of ORGQR.
    Memory consumption linear in matrix rank.
    Intended for use only within the test suite.
    """
    if checks:
        assert torch.is_tensor(v) and torch.is_tensor(tau)
        assert v.dim() == 2 and tau.dim() == 1
        assert v.shape[0] >= v.shape[1]
        assert v.shape[1] == tau.numel()
        assert v.device == tau.device
    out = torch.eye(*v.shape, dtype=v.dtype, device=v.device)
    for i in reversed(range(tau.numel())):
        t = tau[i]
        u = v[:, i].unsqueeze(1)
        out = out - (t * u.T).mm(out) * u
    return out


def reference_orgqr_batch(v, tau, checks=True):
    """
    Unoptimized differentiable implementation of batched ORGQR.
    Memory consumption linear in matrix rank.
    Intended for use only within the test suite.
    """
    if checks:
        assert torch.is_tensor(v) and torch.is_tensor(tau)
        assert v.dim() == 3 and tau.dim() == 2
        assert v.shape[1] >= v.shape[2]
        assert v.shape[2] == tau.shape[1]
        assert v.device == tau.device
    out = torch.eye(*v.shape[1:], dtype=v.dtype, device=v.device).unsqueeze(0).repeat(v.shape[0], 1, 1)
    for i in reversed(range(tau.shape[1])):
        t = tau[:, i].view(-1, 1, 1)
        u = v[:, :, i].unsqueeze(2)
        out = out - (t * u).permute(0, 2, 1).bmm(out) * u
    return out


def pytorch_orgqr_with_roundtrip(v, tau, checks=True):
    if checks:
        assert torch.is_tensor(v) and torch.is_tensor(tau)
        assert v.dim() == 2 and tau.dim() == 1
        assert v.shape[0] >= v.shape[1]
        assert v.shape[1] == tau.numel()
        assert v.device == tau.device
    if v.device == 'cpu':
        return torch.orgqr(v, tau)
    device = v.device
    v, tau = v.cpu(), tau.cpu()
    out = torch.orgqr(v, tau).to(device)
    return out
