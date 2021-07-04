import os
import sys
from subprocess import STDOUT, check_output, TimeoutExpired

import numpy as np
import torch
from packaging import version
from tqdm import tqdm


def resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


def benchmark(method, b, d, r, repeats, timeout_sec=None):
    assert method in ('hh_ours', 'hh_ours_64', 'hh_pt', 'mexp')
    cwd = os.path.dirname(__file__)
    cmd = [
        sys.executable, '-m', 'benchmark_one',
        '--method', method,
        '--repeats', str(repeats),
        '--b', str(b),
        '--d', str(d),
        '--r', str(r),
    ]
    try:
        output = check_output(cmd, stderr=STDOUT, timeout=timeout_sec, cwd=cwd)
        if output == b'OOM\n':
            print('OOM', method, b, d, r)
            return -np.inf, np.nan
        ms, err = output.decode("utf-8").strip('\n').split(' ')
        return float(ms), float(err)
    except TimeoutExpired:
        print('Timeout', method, b, d, r)
        return np.inf, np.nan


def benchmark_all(NB, ND, NR, repeats):
    assert torch.cuda.is_available(), 'CUDA must be available for this benchmark'
    assert version.parse(torch.__version__) >= version.parse('1.9'), \
        'This benchmark requires PyTorch>=1.9, which includes both torch.matrix_exp and ' \
        'torch.linalg.householder_product'

    exp_ms = torch.full((NB, ND, NR), fill_value=np.nan, dtype=torch.float32)
    exp_err = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)
    hh_ours_ms = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)
    hh_ours_err = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)
    hh_ours64_ms = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)
    hh_ours64_err = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)
    hh_pt_ms = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)
    hh_pt_err = torch.full_like(exp_ms, fill_value=np.nan, dtype=torch.float32)

    with tqdm(total=NB * (ND * NR - (NR * (NR - 1) // 2))) as pbar:
        for idb in reversed(range(NB)):
            b = 2 ** idb
            for idd in reversed(range(ND)):
                d = 2 ** idd
                for idr in reversed(range(NR)):
                    r = 2 ** idr
                    if r > d:
                        continue
                    bdr = b * d * r
                    if bdr >= 2 ** 24:
                        repeats = 1
                    timeout_sec = 600  # 10 min should be enough for everything
                    hh_ours_ms[idb, idd, idr], hh_ours_err[idb, idd, idr] = \
                        benchmark('hh_ours', b, d, r, repeats, timeout_sec=timeout_sec)
                    hh_ours64_ms[idb, idd, idr], hh_ours64_err[idb, idd, idr] = \
                        benchmark('hh_ours_64', b, d, r, repeats, timeout_sec=timeout_sec)
                    exp_ms[idb, idd, idr], exp_err[idb, idd, idr] = \
                        benchmark('mexp', b, d, r, repeats, timeout_sec=timeout_sec)
                    hh_pt_ms[idb, idd, idr], hh_pt_err[idb, idd, idr] = \
                        benchmark('hh_pt', b, d, r, repeats, timeout_sec=timeout_sec)
                    pbar.update(1)

    return exp_ms, exp_err, hh_ours_ms, hh_ours_err, hh_ours64_ms, hh_ours64_err, hh_pt_ms, hh_pt_err


if __name__ == '__main__':
    NB, ND, NR, repeats = 12, 16, 16, 3

    benchmark_results = resolve('benchmark_fwdbwd.pth')
    exp_ms, exp_err, hh_ours_ms, hh_ours_err, hh_ours64_ms, hh_ours64_err, hh_pt_ms, hh_pt_err = \
        benchmark_all(NB, ND, NR, repeats)
    state = {
        'header': (NB, ND, NR),
        'tensors': (exp_ms, exp_err, hh_ours_ms, hh_ours_err, hh_ours64_ms, hh_ours64_err, hh_pt_ms, hh_pt_err)
    }
    torch.save(state, benchmark_results)
