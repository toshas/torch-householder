import sys
import unittest

import torch
from tqdm import tqdm

from torch_householder import torch_householder_orgqr as orgqr_hh
from tests.reference_orgqr import reference_orgqr as orgqr_rf
from tests.reference_orgqr import pytorch_orgqr_with_roundtrip as orgqr_pt
from tests.reference_orgqr import reference_orgqr_batch


def eye3d(b, d, r, device=None):
    return torch.eye(d, r, device=device).unsqueeze(0).repeat(b, 1, 1)


def generate_param(d, r, b=None, cuda=False):
    if b is None:
        param = torch.randn(d, r).tril(diagonal=-1) + torch.eye(d, r)
    else:
        param = torch.randn(b, d, r).tril(diagonal=-1) + eye3d(b, d, r)
    if cuda:
        param = param.cuda()
    return param


def compute_orgqr_tau(param, eps=1e-12):
    if param.dim() not in (2, 3):
        raise ValueError(f'Invalid argument dimensions: param={param.dim()}')
    return 2 / (param * param).sum(dim=(param.dim() - 2)).clamp_min(eps)


def tensor_diff(a, b):
    return (a - b).abs().max().item()


class TestORGQR(unittest.TestCase):

    #
    # 2d tau
    #

    def _test_orgqr_2d_tau(self, d, r, cuda=False, eps_out=1e-5, eps_grad_param=1e-4, eps_grad_tau=1e-4):
        assert d >= r > 0
        param = torch.nn.Parameter(generate_param(d, r, cuda=cuda))
        tau = torch.nn.Parameter(compute_orgqr_tau(param).detach())
        eye = torch.eye(d, r)
        eye_r = torch.eye(r)
        if cuda:
            eye = eye.cuda()
            eye_r = eye_r.cuda()

        param64 = torch.nn.Parameter(param.double())
        tau64 = torch.nn.Parameter(tau.double())

        out_pt = orgqr_pt(param, tau)

        out_rf64 = orgqr_rf(param64, tau64)
        out_rf64.sum().backward()
        grad_rf_64_param = param64.grad.clone()
        grad_rf_64_tau = tau64.grad.clone()

        out_hh = orgqr_hh(param, tau, eye=eye)
        out_hh.sum().backward()
        grad_hh_param = param.grad.clone()
        grad_hh_tau = tau.grad.clone()

        dist_eye_rf64 = tensor_diff(out_rf64.T.mm(out_rf64), eye_r.double())
        self.assertLess(dist_eye_rf64, eps_out, f'd={d} r={r} dist_eye_rf64={dist_eye_rf64} eps={eps_out}')

        err_out_pt_rf64 = tensor_diff(out_pt.double(), out_rf64)
        err_out_hh_rf64 = tensor_diff(out_hh.double(), out_rf64)
        self.assertLess(err_out_pt_rf64, eps_out, f'd={d} r={r} err_out_pt_rf64={err_out_pt_rf64} eps={eps_out}')
        self.assertLess(err_out_hh_rf64, eps_out, f'd={d} r={r} err_out_hh_rf64={err_out_hh_rf64} eps={eps_out}')

        err_grad_hh_param_rf64 = tensor_diff(grad_hh_param.double(), grad_rf_64_param)
        err_grad_hh_tau_rf64 = tensor_diff(grad_hh_tau.double(), grad_rf_64_tau)
        self.assertLess(err_grad_hh_param_rf64, eps_grad_param, f'd={d} r={r} err_grad_hh_param_rf64={err_grad_hh_param_rf64} eps={eps_grad_param}')
        self.assertLess(err_grad_hh_tau_rf64, eps_grad_tau, f'd={d} r={r} err_grad_hh_tau_rf64={err_grad_hh_tau_rf64} eps={eps_grad_tau}')

    def test_orgqr_2d_tau_cuda(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_2d_tau_cuda', file=sys.stderr)
            return
        torch.manual_seed(2021)
        Nmax, Rmax = 100, 10
        with tqdm(total=Nmax * Rmax - (Rmax * (Rmax - 1) // 2), desc='test_orgqr_2d_tau_cuda') as pbar:
            for i in range(1, Nmax+1):
                for j in range(1, Rmax+1):
                    if j > i:
                        continue
                    with self.subTest(i=i, j=j):
                        self._test_orgqr_2d_tau(i, j, cuda=True)
                    pbar.update(1)

    def test_orgqr_2d_tau_cpu(self):
        torch.manual_seed(2021)
        Nmax, Rmax = 100, 10
        with tqdm(total=Nmax * Rmax - (Rmax * (Rmax - 1) // 2), desc='test_orgqr_2d_tau_cpu') as pbar:
            for i in range(1, Nmax+1):
                for j in range(1, Rmax+1):
                    if j > i:
                        continue
                    with self.subTest(i=i, j=j):
                        self._test_orgqr_2d_tau(i, j, cuda=False)
                    pbar.update(1)

    def test_orgqr_2d_tau_cuda_2048_128(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_2d_tau_cuda_2048_128', file=sys.stderr)
            return
        torch.manual_seed(2021)
        self._test_orgqr_2d_tau(2048, 128, cuda=True, eps_grad_tau=1e-2)

    def test_orgqr_2d_tau_cuda_2048_512(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_2d_tau_cuda_2048_512', file=sys.stderr)
            return
        torch.manual_seed(2021)
        self._test_orgqr_2d_tau(2048, 512, cuda=True, eps_grad_tau=1e-2)

    #
    # 3d tau
    #

    def _test_orgqr_3d_tau(self, b, d, r, cuda=False, eps_out=1e-5, eps_grad_param=1e-4, eps_grad_tau=5e-4):
        assert d >= r > 0
        param = torch.nn.Parameter(generate_param(d, r, b=b, cuda=cuda))
        tau = torch.nn.Parameter(compute_orgqr_tau(param).detach())
        eye = eye3d(b, d, r)
        eye_r = eye3d(b, r, r)
        if cuda:
            eye = eye.cuda()
            eye_r = eye_r.cuda()

        param64 = torch.nn.Parameter(param.double())
        tau64 = torch.nn.Parameter(tau.double())

        out_rf64 = reference_orgqr_batch(param64, tau64)
        out_rf64.sum().backward()
        grad_rf_64_param = param64.grad.clone()
        grad_rf_64_tau = tau64.grad.clone()

        out_hh = orgqr_hh(param, tau, eye=eye)
        out_hh.sum().backward()
        grad_hh_param = param.grad.clone()
        grad_hh_tau = tau.grad.clone()

        dist_eye_rf64 = tensor_diff(out_rf64.permute(0, 2, 1).bmm(out_rf64), eye_r.double())
        self.assertLess(dist_eye_rf64, eps_out, f'd={d} r={r} dist_eye_rf64={dist_eye_rf64} eps={eps_out}')

        err_out_hh_rf64 = tensor_diff(out_hh.double(), out_rf64)
        self.assertLess(err_out_hh_rf64, eps_out, f'd={d} r={r} err_out_hh_rf64={err_out_hh_rf64} eps={eps_out}')

        err_grad_hh_param_rf64 = tensor_diff(grad_hh_param.double(), grad_rf_64_param)
        err_grad_hh_tau_rf64 = tensor_diff(grad_hh_tau.double(), grad_rf_64_tau)
        self.assertLess(err_grad_hh_param_rf64, eps_grad_param, f'd={d} r={r} err_grad_hh_param_rf64={err_grad_hh_param_rf64} eps={eps_grad_param}')
        self.assertLess(err_grad_hh_tau_rf64, eps_grad_tau, f'd={d} r={r} err_grad_hh_tau_rf64={err_grad_hh_tau_rf64} eps={eps_grad_tau}')

    def test_orgqr_3d_tau_cuda(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_3d_tau_cuda', file=sys.stderr)
            return
        torch.manual_seed(2021)
        for b in (1, 2, 4, 8, 11, 16):
            for i in range(10, 100, 4):
                for j in range(1, 10):
                    if j > i:
                        continue
                    with self.subTest(i=i, j=j):
                        self._test_orgqr_3d_tau(b, i, j, cuda=True)

    def test_orgqr_3d_tau_cpu(self):
        torch.manual_seed(2021)
        for b in (1, 2, 4, 8, 11, 16):
            for i in range(10, 100, 4):
                for j in range(1, 10):
                    if j > i:
                        continue
                    with self.subTest(i=i, j=j):
                        self._test_orgqr_3d_tau(b, i, j, cuda=False)

    def test_orgqr_3d_tau_cuda_2048_128(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_3d_tau_cuda_2048_128', file=sys.stderr)
            return
        torch.manual_seed(2021)
        self._test_orgqr_3d_tau(9, 2048, 128, cuda=True, eps_grad_tau=1e-2)

    def test_orgqr_3d_tau_cuda_2048_512(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_3d_tau_cuda_2048_512', file=sys.stderr)
            return
        torch.manual_seed(2021)
        self._test_orgqr_3d_tau(2, 2048, 512, cuda=True, eps_grad_tau=1e-2)

    #
    # 2d no-tau
    #

    def _test_orgqr_2d_notau(self, d, r, cuda=False, eps_out=1e-5, eps_grad_param=1e-4):
        assert d >= r > 0
        param_val = generate_param(d, r, cuda=cuda)

        eye = torch.eye(d, r)
        if cuda:
            eye = eye.cuda()

        param = torch.nn.Parameter(param_val)
        tau = compute_orgqr_tau(param)
        out_hh_tau = orgqr_hh(param, tau, eye=eye)
        out_hh_tau.sum().backward()
        grad_tau_param = param.grad.clone()

        param = torch.nn.Parameter(param_val)
        out_hh_notau = orgqr_hh(param, eye=eye)
        out_hh_notau.sum().backward()
        grad_notau_param = param.grad.clone()

        err_out = tensor_diff(out_hh_tau, out_hh_notau)
        err_grad = tensor_diff(grad_tau_param, grad_notau_param)
        self.assertLess(err_out, eps_out, f'd={d} r={r} err_out={err_out} eps={eps_out}')
        self.assertLess(err_grad, eps_grad_param, f'd={d} r={r} err_grad={err_grad} eps={eps_grad_param}')

    def test_orgqr_2d_notau_cuda_2048_1024(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_2d_notau_cuda_2048_1024', file=sys.stderr)
            return
        torch.manual_seed(2021)
        self._test_orgqr_2d_notau(2048, 1024, cuda=True)

    #
    # 3d no-tau no-eye
    #

    def _test_orgqr_3d_notau_noeye(self, b, d, r, cuda=False, eps_out=1e-5, eps_grad_param=1e-4):
        assert d >= r > 0
        param_val = generate_param(d, r, b=b, cuda=cuda)

        param = torch.nn.Parameter(param_val)
        tau = compute_orgqr_tau(param)
        out_hh_tau = orgqr_hh(param, tau)
        out_hh_tau.sum().backward()
        grad_tau_param = param.grad.clone()

        param = torch.nn.Parameter(param_val)
        out_hh_notau = orgqr_hh(param)
        out_hh_notau.sum().backward()
        grad_notau_param = param.grad.clone()

        err_out = tensor_diff(out_hh_tau, out_hh_notau)
        err_grad = tensor_diff(grad_tau_param, grad_notau_param)
        self.assertLess(err_out, eps_out, f'd={d} r={r} err_out={err_out} eps={eps_out}')
        self.assertLess(err_grad, eps_grad_param, f'd={d} r={r} err_grad={err_grad} eps={eps_grad_param}')

    def test_orgqr_3d_notau_noeye_cuda_2048_1024(self):
        if not torch.cuda.is_available():
            print('Skipping test_orgqr_3d_notau_noeye_cuda_2048_1024', file=sys.stderr)
            return
        torch.manual_seed(2021)
        self._test_orgqr_3d_notau_noeye(2, 2048, 512, cuda=True)


if __name__ == '__main__':
    unittest.main()
