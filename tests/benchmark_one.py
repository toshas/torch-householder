import argparse
import time
from functools import partial

import torch
from packaging import version

from torch_householder import torch_householder_orgqr

assert torch.cuda.is_available(), 'CUDA must be available for this benchmark'
assert version.parse(torch.__version__) >= version.parse('1.9'), \
    'This benchmark requires PyTorch>=1.9, which includes both torch.matrix_exp and torch.linalg.householder_product'


torch.manual_seed(0)


def tensor_diff(a, b):
    return (a - b).abs().max().item()


def benchmark_hh_ours(b, d, r, repeats, do_backward=True, dtype=torch.float32):
    assert d >= r
    eye = torch.eye(d, r, device='cuda', dtype=dtype).unsqueeze(0).repeat(b, 1, 1)

    start_sec = None
    discrepancies = []

    for i in range(repeats + 1):
        if i == 1:
            torch.cuda.synchronize()
            start_sec = time.monotonic()

        param = torch.randn(b, d, r, device='cuda', dtype=dtype).tril(diagonal=-1) + eye
        param = torch.nn.Parameter(param)

        out = torch_householder_orgqr(param, eye=eye)

        if do_backward:
            loss = out.abs().sum()
            loss.backward()

        if i >= 1:
            with torch.no_grad():
                discrepancies.append(tensor_diff(out.permute(0, 2, 1).bmm(out), eye[:, :r, :r]))

    torch.cuda.synchronize()
    end_sec = time.monotonic()

    avg_ms = (end_sec - start_sec) * 1000 / repeats
    avg_err = torch.tensor(discrepancies).mean().item()
    return avg_ms, avg_err


def benchmark_hh_pytorch(b, d, r, repeats, do_backward=True):
    assert d >= r
    eye = torch.eye(d, r, device='cuda').unsqueeze(0).repeat(b, 1, 1)

    start_sec = None
    discrepancies = []

    for i in range(repeats + 1):
        if i == 1:
            torch.cuda.synchronize()
            start_sec = time.monotonic()

        param = torch.randn(b, d, r, device='cuda').tril(diagonal=-1) + eye
        param = torch.nn.Parameter(param)
        tau = 2 / (param * param).sum(dim=(param.dim() - 2))

        out = torch.linalg.householder_product(param, tau)

        if do_backward:
            loss = out.abs().sum()
            loss.backward()

        if i >= 1:
            with torch.no_grad():
                discrepancies.append(tensor_diff(out.permute(0, 2, 1).bmm(out), eye[:, :r, :r]))

    torch.cuda.synchronize()
    end_sec = time.monotonic()

    avg_ms = (end_sec - start_sec) * 1000 / repeats
    avg_err = torch.tensor(discrepancies).mean().item()
    return avg_ms, avg_err


def benchmark_mexp(b, d, r, repeats, do_backward=True):
    assert d >= r
    eye = torch.eye(d, device='cuda').unsqueeze(0).repeat(b, 1, 1)

    start_sec = None
    discrepancies = []

    for i in range(repeats + 1):
        if i == 1:
            torch.cuda.synchronize()
            start_sec = time.monotonic()

        param = torch.randn(b, d, d, device='cuda').tril(diagonal=-1)
        param = param - param.permute(0, 2, 1)  # skew-symmetric
        param = torch.nn.Parameter(param)

        out = torch.matrix_exp(param)[:, :, :r]

        if do_backward:
            loss = out.abs().sum()
            loss.backward()

        if i >= 1:
            with torch.no_grad():
                discrepancies.append(tensor_diff(out.permute(0, 2, 1).bmm(out), eye[:, :r, :r]))

    torch.cuda.synchronize()
    end_sec = time.monotonic()

    avg_ms = (end_sec - start_sec) * 1000 / repeats
    avg_err = torch.tensor(discrepancies).mean().item()
    return avg_ms, avg_err


def benchmark(args):
    benchmark = {
        'hh_ours': benchmark_hh_ours,
        'hh_ours_64': partial(benchmark_hh_ours, dtype=torch.float64),
        'hh_pt': benchmark_hh_pytorch,
        'mexp': benchmark_mexp,
    }[args.method]
    try:
        avg_ms, avg_err = benchmark(args.b, args.d, args.r, args.repeats)
    except RuntimeError as e:
        if not 'CUDA out of memory' in str(e):
            raise e
        return 'OOM'
    return f'{avg_ms} {avg_err}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, type=str, choices=('hh_ours', 'hh_ours_64', 'hh_pt', 'mexp'))
    parser.add_argument('--repeats', required=True, type=int)
    parser.add_argument('--b', required=True, type=int)
    parser.add_argument('--d', required=True, type=int)
    parser.add_argument('--r', required=True, type=int)
    args = parser.parse_args()
    out = benchmark(args)
    print(out)
