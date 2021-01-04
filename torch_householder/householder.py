import os

import torch
from torch.utils.cpp_extension import load

VERBOSE = False


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


try:
    import torch_householder_cpp
except ImportError:
    if VERBOSE:
        print('Falling back to JIT compiling torch_householder_cpp')
    torch_householder_cpp = load(
        name='torch_householder_cpp',
        sources=[
            _resolve('householder.cpp'),
        ],
        verbose=VERBOSE,
    )


class OpORGQRNormalized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        m = torch_householder_cpp.orgqr_normalized_forward(param)
        ctx.save_for_backward(m, param)
        return m

    @staticmethod
    def backward(ctx, grad_out):
        m, param = ctx.saved_tensors
        grad_param = torch_householder_cpp.orgqr_normalized_backward(grad_out, m, param)
        return grad_param


class OpORGQRNormalizedEye(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, eye):
        m = torch_householder_cpp.orgqr_normalized_eye_forward(param, eye)
        ctx.save_for_backward(m, param)
        return m

    @staticmethod
    def backward(ctx, grad_out):
        m, param = ctx.saved_tensors
        grad_param = torch_householder_cpp.orgqr_normalized_backward(grad_out, m, param)
        return grad_param, None


def torch_householder_orgqr(param, tau=None, eye=None, eps=1e-12):
    if param.dim() not in (2, 3):
        raise ValueError(f'Invalid argument shapes: param={param.dim()}, tau={tau.dim()}')
    if tau is not None:
        if param.dim() != tau.dim() + 1:
            raise ValueError(f'Invalid argument shapes: param={param.dim()}, tau={tau.dim()}')
        if param.dim() == 3 and (param.shape[0] != tau.shape[0] or param.shape[2] != tau.shape[1]) or \
                param.dim() == 2 and param.shape[1] != tau.shape[0]:
            raise ValueError(f'Incompatible argument shapes: param={param.dim()}, tau={tau.dim()}')
        half_tau_sqrt = (tau.unsqueeze(-2) * 0.5).sqrt()
        param_normalized = param * half_tau_sqrt
    else:
        param_normalized = param / torch.linalg.norm(param, dim=param.dim()-2, keepdim=True).clamp(min=eps)
    restore_2d = False
    if eye is not None:
        if eye.shape != param.shape or eye.device != param.device:
            raise ValueError(f'Invalid parameter shape or device placement')
        if param_normalized.dim() == 2:
            restore_2d = True
            param_normalized = param_normalized.unsqueeze(0)
            eye = eye.unsqueeze(0)
        out = OpORGQRNormalizedEye.apply(param_normalized, eye)
    else:
        if param_normalized.dim() == 2:
            restore_2d = True
            param_normalized = param_normalized.unsqueeze(0)
        out = OpORGQRNormalized.apply(param_normalized)
    if restore_2d:
        out = out.squeeze(0)
    return out
