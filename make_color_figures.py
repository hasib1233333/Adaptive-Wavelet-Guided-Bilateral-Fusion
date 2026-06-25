"""
Figures for the color-image extension (Kodak24 + McMaster).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method_color import (
    awgb_denoise_color, baseline_gaussian_color, baseline_median_color,
    baseline_bilateral_color, baseline_nlm_color, baseline_wavelet_only_color,
    baseline_bm3d_color, estimate_noise_sigma_color
)
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'axes.linewidth': 0.8,
                      'figure.dpi': 100, 'savefig.dpi': 300})

FIG_DIR = '/home/claude/awgb/figures_v2'
RES_DIR = '/home/claude/awgb/results'
DATA_DIR = '/home/claude/awgb/data'
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RES_DIR, 'color_results.json')) as f:
    RESULTS = json.load(f)

METHOD_ORDER = ['Gaussian', 'Median', 'Bilateral', 'NLM', 'WaveletOnly', 'BM3D', 'AWGB (Proposed)']
METHOD_LABELS = {m: m for m in METHOD_ORDER}
METHOD_LABELS['WaveletOnly'] = 'Wavelet-only'
METHOD_LABELS['AWGB (Proposed)'] = 'AWGB (Proposed)'
COLORS = {
    'Gaussian': '#9e9e9e', 'Median': '#7a7a7a', 'Bilateral': '#4a90a4',
    'NLM': '#c77b3a', 'WaveletOnly': '#7e6b9e', 'BM3D': '#2e7d4f',
    'AWGB (Proposed)': '#c0392b'
}


def dataset_of(name):
    return 'Kodak24' if name.startswith('Kodak24') else 'McMaster'


# ---------------- Figure: average rank bar chart (color) ----------------
def fig_color_rank():
    methods = ['Gaussian', 'Median', 'Bilateral', 'NLM', 'WaveletOnly', 'AWGB (Proposed)']
    ranks = {m: [] for m in methods}
    for img in RESULTS:
        for sigma in RESULTS[img]:
            sub = {m: RESULTS[img][sigma][m]['psnr_mean'] for m in methods}
            order = sorted(sub.items(), key=lambda x: -x[1])
            for i, (m, v) in enumerate(order):
                ranks[m].append(i + 1)
    avg_ranks = {m: np.mean(ranks[m]) for m in methods}
    order = sorted(avg_ranks.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    names = [METHOD_LABELS[m] for m, _ in order]
    vals = [v for _, v in order]
    bar_colors = [COLORS[m] for m, _ in order]
    bars = ax.barh(names, vals, color=bar_colors, edgecolor='black', linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2, f'{v:.2f}', va='center', fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('Average PSNR Rank (1 = best, lightweight methods only)')
    ax.set_title('Mean Ranking on Color Images (Kodak24+McMaster)\n42 images $\\times$ 3 noise levels (n=126)', fontsize=10)
    ax.grid(alpha=0.3, axis='x', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_color_rank.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig_color_rank.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig_color_rank")


# ---------------- Figure: visual comparison on a Kodak24 image ----------------
def fig_color_visual():
    clean = np.array(Image.open(os.path.join(DATA_DIR, 'Kodak24', 'kodim23.png')).convert('RGB'), dtype=np.float64)
    sigma = 25
    rng = np.random.RandomState(10000 * sigma + 0)
    noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255)
    sigma_est = estimate_noise_sigma_color(noisy)

    outputs = {
        'Noisy Input': noisy,
        'Gaussian': baseline_gaussian_color(noisy),
        'Median': baseline_median_color(noisy),
        'Bilateral': baseline_bilateral_color(noisy),
        'NLM': baseline_nlm_color(noisy),
        'Wavelet-only': baseline_wavelet_only_color(noisy),
        'BM3D': baseline_bm3d_color(noisy, sigma_est),
        'AWGB (Proposed)': awgb_denoise_color(noisy)[0],
        'Ground Truth': clean,
    }

    fig, axes = plt.subplots(3, 3, figsize=(11, 9.0))
    order = ['Noisy Input', 'Gaussian', 'Median', 'Bilateral', 'NLM', 'Wavelet-only',
             'BM3D', 'AWGB (Proposed)', 'Ground Truth']
    for ax, name in zip(axes.flat, order):
        img = np.clip(outputs[name], 0, 255).astype(np.uint8)
        ax.imshow(img)
        is_proposed = 'Proposed' in name
        ax.set_title(name, fontsize=12.5, fontweight='bold' if is_proposed else 'normal',
                     color='#b3221c' if is_proposed else '#1a1a1a')
        if name not in ('Ground Truth',):
            p = psnr_fn(clean, img.astype(np.float64), data_range=255)
            s = ssim_fn(clean, img.astype(np.float64), data_range=255, channel_axis=2)
            label = f'PSNR {p:.2f} dB | SSIM {s:.3f}'
            ax.text(0.5, 0.02, label, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=9.5, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.28', facecolor='#b3221c' if is_proposed else '#2c2c2c',
                              edgecolor='none', alpha=0.88))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(FIG_DIR, 'fig_color_visual.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig_color_visual.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig_color_visual")


if __name__ == '__main__':
    fig_color_rank()
    fig_color_visual()
