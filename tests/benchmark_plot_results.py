import copy
import os

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from matplotlib.lines import Line2D

COLOR_BOTH_OOM = (255, 255, 255)
COLOR_ONLY_CMP_OOM = (22, 151, 183)
COLOR_CMP_TIMEOUT = (172, 0, 172)
COLOR_CMP_OOM_OUR_TIMEOUT = (192, 192, 0)


def resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)


def plot_many(
        x, fname, map_z=None, map_y=None, map_x=None, title=None, title_s=None,
        ncol=4, plt_sz_one=2.5, log_norm=False, clamp=1e2,
        cmp_method=None, our_method=None, have_timeout=False
):
    assert type(x) is np.ndarray and len(x.shape) == 4 and x.shape[0] % ncol == 0

    if log_norm:
        x_min = 1.0 / clamp
        x_max = 1.0 * clamp
        cls_normalize = matplotlib.colors.LogNorm
    else:
        x_min = -1.0 * clamp
        x_max = 1.0 * clamp
        cls_normalize = matplotlib.colors.Normalize

    b_h, b_w = x.shape[0] // ncol, ncol

    cmap = matplotlib.cm.get_cmap('coolwarm_r')
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
        axs[i, j].imshow(x[idz])
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

    fig.supxlabel('Matrix width', x=0.45)
    fig.supylabel('Matrix height', x=0.09)

    im = matplotlib.cm.ScalarMappable(cmap=cmap, norm=cm_normalizer)
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), location='right')
    cb.ax.text(0.55, 0.01, f'{cmp_method} is better', rotation=90, color='white', weight='bold', transform=cb.ax.transAxes, va='bottom', ha='center')
    cb.ax.text(0.55, 0.99, f'{our_method} is better', rotation=90, color='white', weight='bold', transform=cb.ax.transAxes, va='top', ha='center')

    legend_options = dict(marker='s', color='black', markersize=10, linewidth=0)
    legend_elements = [
        Line2D([0], [0], label=f'{cmp_method} out of memory, {our_method} completed',
               markerfacecolor=tuple(x / 255 for x in COLOR_ONLY_CMP_OOM), **legend_options),
        Line2D([0], [0], label=f'{cmp_method} out of memory, {our_method} out of memory',
               markerfacecolor=tuple(x / 255 for x in COLOR_BOTH_OOM), **legend_options),
    ]
    if have_timeout:
        legend_elements += [
            Line2D([0], [0], label=f'{cmp_method} timed out, {our_method} completed',
                   markerfacecolor=tuple(x / 255 for x in COLOR_CMP_TIMEOUT), **legend_options),
            Line2D([0], [0], label=f'{cmp_method} out of memory, {our_method} timed out',
                   markerfacecolor=tuple(x / 255 for x in COLOR_CMP_OOM_OUR_TIMEOUT), **legend_options),
        ]
    legend_elements += [
        Line2D([0], [0], label='Fat matrices area',
               markerfacecolor='black', **legend_options),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=[
        0.07,
        axs[-1, -1].get_position(True).y0 - 0.175,
        0.8,
        1,
    ], bbox_transform=fig.transFigure)

    fig.savefig(f'{fname}.png', bbox_inches='tight')


def extract_masks_one(ms):
    mask_oom = ms == -np.inf
    mask_timeout = ms == np.inf
    return mask_oom, mask_timeout


def extract_masks(cmp_ms, our_ms):
    mask_oom_cmp, mask_timeout_cmp = extract_masks_one(cmp_ms)
    mask_oom_our, mask_timeout_our = extract_masks_one(our_ms)
    mask_oom_both = mask_oom_cmp & mask_oom_our
    mask_oom_only_cmp = mask_oom_cmp & (~mask_oom_our)
    mask_oom_only_our = mask_oom_our & (~mask_oom_cmp)
    mask_timeout_both = mask_timeout_cmp & mask_timeout_our
    mask_timeout_only_cmp = mask_timeout_cmp & (~mask_timeout_our)
    mask_timeout_only_our = mask_timeout_our & (~mask_timeout_cmp)
    mask_oom_cmp_timeout_our = mask_timeout_only_our & mask_oom_only_cmp
    mask_oom_our_timeout_cmp = mask_timeout_only_cmp & mask_oom_only_our
    # in case any of these checks fail, new categories/colors need to be added to the plots
    assert torch.all(~mask_oom_only_our)
    assert torch.all(~mask_timeout_only_our | mask_oom_only_cmp)
    assert torch.all(~mask_oom_our_timeout_cmp)
    assert torch.all(~mask_timeout_both)
    return mask_oom_both, mask_oom_only_cmp, mask_timeout_cmp, mask_oom_cmp_timeout_our


def make_rgb(x, mask_oom_both, mask_oom_only_cmp, mask_timeout_cmp, mask_oom_cmp_timeout_our, log_norm, clamp):
    assert torch.is_tensor(x) and x.dim() == 3

    if log_norm:
        x_min = 1.0 / clamp
        x_max = 1.0 * clamp
        x[(x > x_max) & (x < np.inf)] = x_max
        x[(x < x_min) & (x > -np.inf)] = x_min
        cls_normalize = matplotlib.colors.LogNorm
    else:
        x_min = -1.0 * clamp
        x_max = 1.0 * clamp
        x[(x > x_max) & (x < np.inf)] = x_max
        x[(x < x_min) & (x > -np.inf)] = x_min
        cls_normalize = matplotlib.colors.Normalize

    cm_normalizer = cls_normalize(x_min, x_max)

    cmap = matplotlib.cm.get_cmap('coolwarm_r')
    cmap = copy.copy(cmap)
    cmap.set_bad(color='black')
    cmap.set_over(color='black')
    cmap.set_under(color='black')

    x = x.numpy()
    x_shape = x.shape
    x = np.ravel(x)
    x = cm_normalizer(x)
    x = np.reshape(x, x_shape)

    x = cmap(x, bytes=False)
    x = x[:, :, :, 0:3]
    x = (255 * x).astype(np.uint8)

    x[mask_oom_both] = COLOR_BOTH_OOM
    x[mask_oom_only_cmp] = COLOR_ONLY_CMP_OOM
    x[mask_timeout_cmp] = COLOR_CMP_TIMEOUT
    x[mask_oom_cmp_timeout_our] = COLOR_CMP_OOM_OUR_TIMEOUT

    return x


def make_plot_speed(cmp_ms, cmp_name, our_ms, our_name):
    mask_oom_both, mask_only_their_oom, mask_their_timeout, mask_oom_cmp_timeout_our = extract_masks(cmp_ms, our_ms)
    var = cmp_ms / our_ms
    log_norm = True
    clamp = 10
    title = f'Run time ratio of {cmp_name} over {our_name}'
    fname = f'benchmark_speed_{cmp_name}_vs_{our_name}'
    var = make_rgb(var, mask_oom_both, mask_only_their_oom, mask_their_timeout, mask_oom_cmp_timeout_our,
                   log_norm=log_norm, clamp=clamp)
    plot_many(var, fname, lambda z: 2 ** z, lambda y: 2 ** y, lambda x: 2 ** x, title, 'Batch size',
              log_norm=log_norm, clamp=clamp, cmp_method=cmp_name, our_method=our_name,
              have_timeout=torch.any(mask_their_timeout | mask_oom_cmp_timeout_our).item())


def make_plot_err_diff(cmp_ms, cmp_err, cmp_name, our_ms, our_err, our_name):
    mask_oom_both, mask_only_their_oom, mask_their_timeout, mask_oom_cmp_timeout_our = \
        extract_masks(cmp_ms, our_ms)
    var = cmp_err - our_err
    log_norm = False
    clamp = 1e-6
    title = f'Absolute errors difference between {cmp_name} and {our_name}'
    fname = f'benchmark_error_{cmp_name}_vs_{our_name}'
    var = make_rgb(
        var, mask_oom_both, mask_only_their_oom, mask_their_timeout, mask_oom_cmp_timeout_our,
        log_norm=log_norm, clamp=clamp
    )
    plot_many(var, fname, lambda z: 2 ** z, lambda y: 2 ** y, lambda x: 2 ** x, title, 'Batch size',
              log_norm=log_norm, clamp=clamp, cmp_method=cmp_name, our_method=our_name,
              have_timeout=torch.any(mask_their_timeout | mask_oom_cmp_timeout_our).item())


if __name__ == '__main__':
    NB, ND, NR, repeats = 12, 16, 16, 3

    benchmark_results = resolve('benchmark_fwdbwd.pth')
    assert os.path.exists(benchmark_results)

    state = torch.load(benchmark_results)
    nb, nd, nr = state['header']
    assert NB == nb and ND == nd and NR == nr
    exp_ms, exp_err, hh_ours_ms, hh_ours_err, hh_ours64_ms, hh_ours64_err, hh_pt_ms, hh_pt_err = state['tensors']

    make_plot_speed(exp_ms, 'MEXP', hh_ours_ms, 'HH_OURS')
    make_plot_err_diff(exp_ms, exp_err, 'MEXP', hh_ours_ms, hh_ours_err, 'HH_OURS')

    make_plot_speed(hh_pt_ms, 'HH_PT', hh_ours_ms, 'HH_OURS')
    make_plot_err_diff(hh_pt_ms, hh_pt_err, 'HH_PT', hh_ours_ms, hh_ours_err, 'HH_OURS')

    make_plot_speed(hh_pt_ms, 'HH_PT', hh_ours64_ms, 'HH_OURS_64')
    make_plot_err_diff(hh_pt_ms, hh_pt_err, 'HH_PT', hh_ours64_ms, hh_ours64_err, 'HH_OURS_64')
