import copy
import gc
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from matplotlib.lines import Line2D
from tqdm import tqdm

from torch_householder import torch_householder_orgqr as orgqr


def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


def tensor_diff(a, b):
    return (a - b).abs().max().item()


def benchmark_orgqr(b, d, r, repeats=30, do_backward=True):
    assert d >= r
    if not torch.cuda.is_available():
        raise RuntimeError('Benchmaring requires CUDA')
    eye = torch.eye(d, r, device='cuda:0').unsqueeze(0).repeat(b, 1, 1)

    start_sec = None
    discrepancies = []

    for i in range(repeats + 1):
        if i == 1:
            torch.cuda.synchronize()
            start_sec = time.monotonic()

        param = torch.randn(b, d, r, device='cuda:0').tril(diagonal=-1) + eye
        param = torch.nn.Parameter(param)

        out = orgqr(param, eye=eye)

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


def benchmark_mexp(b, d, r, repeats=30, do_backward=True):
    assert d >= r
    if not torch.cuda.is_available():
        raise RuntimeError('Benchmaring requires CUDA')
    eye = torch.eye(d, device='cuda:0').unsqueeze(0).repeat(b, 1, 1)

    start_sec = None
    discrepancies = []

    for i in range(repeats + 1):
        if i == 1:
            torch.cuda.synchronize()
            start_sec = time.monotonic()

        param = torch.randn(b, d, d, device='cuda:0').tril(diagonal=-1)
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


def plot_many(
        x, fname, map_z=None, map_y=None, map_x=None, title=None, title_s=None,
        ncol=4, plt_sz_one=2.5, log_norm=False, clamp=1e2
):
    assert torch.is_tensor(x) and x.dim() == 3 and x.shape[0] % ncol == 0

    x[(x > clamp) & (x < np.inf) & (x > -np.inf)] = clamp
    x[(x < 1/clamp) & (x < np.inf) & (x > -np.inf)] = 1/clamp

    x_valid = x[x == x]
    x_valid = x_valid[x_valid < np.inf]
    if log_norm:
        x_valid = x_valid[x_valid > 0]
    x_min, x_max = x_valid.min().item(), x_valid.max().item()
    b_h, b_w = x.shape[0] // ncol, ncol

    cmap = matplotlib.cm.get_cmap('coolwarm_r')
    cmap = copy.copy(cmap)
    cmap.set_bad(color='black')
    cmap.set_over(color='darkturquoise')
    cmap.set_under(color='white')

    if log_norm:
        x_abs_max = max(x_max, 1 / x_min)
        x_min = 1 / x_abs_max
        x_max = x_abs_max
    else:
        x_abs_max = max(x_max, -x_min)
        x_min = -x_abs_max
        x_max = x_abs_max

    for k in range(x.shape[0]):
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                if x[k, i, j] != x[k, i, j]:
                    if i == j == 0:
                        x[k, i, j] = 1.0
                    elif i >= j:
                        x[k, i, j] = -np.inf

    # inf and -inf dont trigger over- and underflow thresholds with LogNorm
    x[x > x_max] = 2 * x_max
    x[x < x_min] = 0.5 * x_min

    if log_norm:
        cls_normalize = matplotlib.colors.LogNorm
    else:
        cls_normalize = matplotlib.colors.Normalize
    cm_normalizer = cls_normalize(x_min, x_max)

    fig, axs = plt.subplots(
        b_h, b_w,
        figsize=(plt_sz_one * (b_w + 1.25), plt_sz_one * b_h),
        gridspec_kw={'hspace': 0.08 * plt_sz_one, 'wspace': 0.01 * plt_sz_one},
    )
    if title is not None:
        fig.suptitle(title, x=0.47)

    for idz in range(x.shape[0]):
        z = map_z(idz) if map_z is not None else idz
        i = idz // ncol
        j = idz % ncol
        axs[i, j].imshow(x[idz], cmap=cmap, norm=cm_normalizer)
        axs[i, j].set_aspect('equal')
        if title_s is not None:
            axs[i, j].set_title(title_s + f': {z}', fontsize=10)
        if i < x.shape[0] // ncol - 1:
            axs[i, j].xaxis.set_visible(False)
        else:
            axs[i, j].xaxis.set_major_locator(ticker.FixedLocator(list(range(x.shape[2]))))
            axs[i, j].set_xticklabels([f'{map_x(ix)}' for ix in range(x.shape[2])], rotation=90, fontsize=6)
        if j > 0:
            axs[i, j].yaxis.set_visible(False)
        else:
            axs[i, j].yaxis.set_major_locator(ticker.FixedLocator(list(range(x.shape[1]))))
            axs[i, j].set_yticklabels([f'{map_y(iy)}' for iy in range(x.shape[1])], fontsize=6)

    im = matplotlib.cm.ScalarMappable(cmap=cmap, norm=cm_normalizer)
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), location='right')
    cb.ax.text(0.15, 0.0125, 'MEXP is better', rotation=90, color='white', weight='bold')
    cb.ax.text(0.15, 20, 'HH is better', rotation=90, color='white', weight='bold')

    legend_options = dict(marker='s', color='black', markersize=10, linewidth=0)
    legend_elements = [
        Line2D([0], [0], label='HH completed, MEXP out of memory', markerfacecolor='darkturquoise', **legend_options),
        Line2D([0], [0], label='Both HH and MEXP out of memory', markerfacecolor='white', **legend_options),
        Line2D([0], [0], label='Fat matrices area', markerfacecolor='black', **legend_options),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=[
        0.07,
        axs[-1, -1].get_position(True).y0 - 0.125,
        0.8,
        1,
    ], bbox_transform=fig.transFigure)  #

    fig.savefig(f'{fname}.png', bbox_inches='tight')


def benchmark_all(NB, ND, NR, repeats, do_backward=True):
    exp_ms = torch.full((NB, ND, NR), fill_value=np.nan, dtype=torch.float32)
    exp_err = torch.full_like(exp_ms, fill_value=np.nan)
    hh_ms = torch.full_like(exp_ms, fill_value=np.nan)
    hh_err = torch.full_like(exp_ms, fill_value=np.nan)

    with tqdm(total=NB * (ND * NR - (NR * (NR - 1) // 2))) as pbar:
        for idb in reversed(range(NB)):
            b = 2 ** idb
            for idd in reversed(range(ND)):
                d = 2 ** idd
                for idr in reversed(range(NR)):
                    r = 2 ** idr
                    if r > d:
                        continue
                    try:
                        exp_ms[idb, idd, idr], exp_err[idb, idd, idr] = benchmark_mexp(b, d, r, repeats, do_backward)
                    except RuntimeError as e:
                        if not 'CUDA out of memory' in str(e):
                            raise e
                        torch.cuda.empty_cache()
                        gc.collect()
                        exp_ms[idb, idd, idr], exp_err[idb, idd, idr] = np.inf, np.inf
                    try:
                        hh_ms[idb, idd, idr], hh_err[idb, idd, idr] = benchmark_orgqr(b, d, r, repeats, do_backward)
                    except RuntimeError as e:
                        if not 'CUDA out of memory' in str(e):
                            raise e
                        torch.cuda.empty_cache()
                        gc.collect()
                        hh_ms[idb, idd, idr], hh_err[idb, idd, idr] = np.inf, np.inf
                    pbar.update(1)

    return exp_ms, exp_err, hh_ms, hh_err


if __name__ == '__main__':
    NB, ND, NR, repeats = 12, 16, 16, 5

    benchmark_results = _resolve('benchmark_fwdbwd.pth')
    if os.path.exists(benchmark_results):
        state = torch.load(benchmark_results)
        nb, nd, nr = state['header']
        assert NB == nb and ND == nd and NR == nr
        exp_ms, exp_err, hh_ms, hh_err = state['tensors']
    else:
        exp_ms, exp_err, hh_ms, hh_err = benchmark_all(NB, ND, NR, repeats)
        state = {'header': (NB, ND, NR), 'tensors': (exp_ms, exp_err, hh_ms, hh_err)}
        torch.save(state, benchmark_results)

    plot_many(
        exp_ms / hh_ms,
        'benchmark_speed',
        lambda z: 2 ** z,
        lambda y: 2 ** y,
        lambda x: 2 ** x,
        'Run time ratio of matrix exponential (MEXP) over Householder (HH) as a function of matrix size',
        'batch size',
        log_norm=True,
    )

    plot_many(
        exp_err / hh_err,
        'benchmark_err',
        lambda z: 2 ** z,
        lambda y: 2 ** y,
        lambda x: 2 ** x,
        'Error ratio of matrix exponential (MEXP) over Householder (HH) as a function of matrix size',
        'batch size',
        log_norm=True,
    )
