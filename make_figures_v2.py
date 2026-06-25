"""
Generate publication figures from the expanded Set12+BSD68 benchmark
(80 images, 3 noise levels, corrected AWGB v2 fusion design).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method import (
    awgb_denoise, baseline_gaussian, baseline_median, baseline_bilateral,
    baseline_nlm, baseline_wavelet_only, baseline_bm3d, estimate_noise_sigma
)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.linewidth': 0.8,
    'figure.dpi': 100, 'savefig.dpi': 300,
})

FIG_DIR = '/home/claude/awgb/figures_v2'
RES_DIR = '/home/claude/awgb/results'
DATA_DIR = '/home/claude/awgb/data'
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RES_DIR, 'expanded_results.json')) as f:
    RESULTS = json.load(f)

METHOD_ORDER = ['Gaussian', 'Median', 'Bilateral', 'NLM', 'WaveletOnly', 'BM3D', 'AWGB (Proposed)']
METHOD_LABELS = {
    'Gaussian': 'Gaussian', 'Median': 'Median', 'Bilateral': 'Bilateral',
    'NLM': 'NLM', 'WaveletOnly': 'Wavelet-only', 'BM3D': 'BM3D',
    'AWGB (Proposed)': 'AWGB (Proposed)'
}
COLORS = {
    'Gaussian': '#9e9e9e', 'Median': '#7a7a7a', 'Bilateral': '#4a90a4',
    'NLM': '#c77b3a', 'WaveletOnly': '#7e6b9e', 'BM3D': '#2e7d4f',
    'AWGB (Proposed)': '#c0392b'
}


def dataset_of(name):
    return 'Set12' if name.startswith('Set12') else 'BSD68'


# ----------------------------------------------------------------------
# Figure 1: PSNR/SSIM vs noise level, averaged across all 80 images
# ----------------------------------------------------------------------
def fig_psnr_ssim_curves():
    sigmas = [15, 25, 50]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for method in METHOD_ORDER:
        psnr_means, ssim_means = [], []
        for sigma in sigmas:
            vals = [RESULTS[img][str(sigma)][method]['psnr_mean'] for img in RESULTS]
            svals = [RESULTS[img][str(sigma)][method]['ssim_mean'] for img in RESULTS]
            psnr_means.append(np.mean(vals))
            ssim_means.append(np.mean(svals))
        style = '-o' if method == 'AWGB (Proposed)' else '--s'
        lw = 2.4 if method == 'AWGB (Proposed)' else 1.3
        ms = 7 if method == 'AWGB (Proposed)' else 5
        axes[0].plot(sigmas, psnr_means, style, label=METHOD_LABELS[method], color=COLORS[method], linewidth=lw, markersize=ms)
        axes[1].plot(sigmas, ssim_means, style, label=METHOD_LABELS[method], color=COLORS[method], linewidth=lw, markersize=ms)
    axes[0].set_xlabel(r'Noise level $\sigma$'); axes[0].set_ylabel('Average PSNR (dB)')
    axes[0].set_title('(a) PSNR vs. Noise Level (n=80 images)'); axes[0].grid(alpha=0.3, linestyle=':'); axes[0].set_xticks(sigmas)
    axes[1].set_xlabel(r'Noise level $\sigma$'); axes[1].set_ylabel('Average SSIM')
    axes[1].set_title('(b) SSIM vs. Noise Level (n=80 images)'); axes[1].grid(alpha=0.3, linestyle=':'); axes[1].set_xticks(sigmas)
    axes[1].legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_psnr_ssim_curves.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig2_psnr_ssim_curves.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig2_psnr_ssim_curves")


# ----------------------------------------------------------------------
# Figure 2: per-dataset bar comparison (Set12 vs BSD68)
# ----------------------------------------------------------------------
def fig_per_dataset_bars():
    methods_lw = ['Gaussian', 'Median', 'Bilateral', 'NLM', 'WaveletOnly', 'AWGB (Proposed)']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, dataset in zip(axes, ['Set12', 'BSD68']):
        means = []
        for m in methods_lw:
            vals = [RESULTS[img][s][m]['psnr_mean'] for img in RESULTS if dataset_of(img) == dataset for s in ['15', '25', '50']]
            means.append(np.mean(vals))
        bar_colors = [COLORS[m] for m in methods_lw]
        bars = ax.bar(range(len(methods_lw)), means, color=bar_colors, edgecolor='black', linewidth=0.6)
        for i, m in enumerate(methods_lw):
            if m == 'AWGB (Proposed)':
                bars[i].set_edgecolor('black'); bars[i].set_linewidth(1.8)
        ax.set_xticks(range(len(methods_lw)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_lw], rotation=30, ha='right', fontsize=9)
        n_imgs = 12 if dataset == 'Set12' else 68
        ax.set_title(f'{dataset} (n={n_imgs} images, avg over $\\sigma\\in\\{{15,25,50\\}}$)', fontsize=10.5)
        ax.grid(alpha=0.3, axis='y', linestyle=':')
    axes[0].set_ylabel('Average PSNR (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig_per_dataset_bars.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig_per_dataset_bars.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig_per_dataset_bars")


# ----------------------------------------------------------------------
# Figure 3: speed-quality tradeoff at sigma=25, averaged over all 80 images
# ----------------------------------------------------------------------
def fig_speed_quality_tradeoff():
    sigma = 25
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    label_offsets = {
        'Gaussian': (10, 6), 'Median': (10, -14), 'Bilateral': (-70, 12),
        'NLM': (10, 6), 'WaveletOnly': (10, -16), 'BM3D': (-45, 10),
        'AWGB (Proposed)': (10, 10),
    }
    for method in METHOD_ORDER:
        psnrs = [RESULTS[img][str(sigma)][method]['psnr_mean'] for img in RESULTS]
        times = [RESULTS[img][str(sigma)][method]['time_ms_mean'] for img in RESULTS]
        avg_psnr, avg_time = np.mean(psnrs), np.mean(times)
        marker = '*' if method == 'AWGB (Proposed)' else 'o'
        size = 340 if method == 'AWGB (Proposed)' else 140
        ax.scatter(avg_time, avg_psnr, s=size, marker=marker, color=COLORS[method],
                   edgecolor='black', linewidth=0.6, zorder=5 if method == 'AWGB (Proposed)' else 3)
        dx, dy = label_offsets[method]
        ax.annotate(METHOD_LABELS[method], (avg_time, avg_psnr), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9.5, fontweight='bold' if method == 'AWGB (Proposed)' else 'normal')
    ax.set_xscale('log')
    ax.set_xlabel('Average runtime per image (ms, log scale, n=80 images)')
    ax.set_ylabel(r'Average PSNR at $\sigma=25$ (dB)')
    ax.set_title('Speed--Quality Tradeoff (Set12 + BSD68)')
    ax.set_xlim(0.3, 80000)
    ax.grid(alpha=0.3, linestyle=':', which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_speed_quality_tradeoff.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig3_speed_quality_tradeoff.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig3_speed_quality_tradeoff")


# ----------------------------------------------------------------------
# Figure 4: pipeline diagram (updated: self-guided bilateral, not joint)
# ----------------------------------------------------------------------
def fig_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.3, 5.0)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#eef3f7', ec='#34495e', fontsize=9.6):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, zorder=3)
        return (x, y, w, h)

    def arrow(p1, p2, lw=1.4, color='#34495e', connectionstyle=None):
        kwargs = dict(arrowstyle='-|>', color=color, lw=lw)
        if connectionstyle:
            kwargs['connectionstyle'] = connectionstyle
        ax.annotate('', xy=p2, xytext=p1, arrowprops=kwargs)

    b_in = box(0.2, 3.7, 1.9, 0.9, 'Noisy Input\nImage $I_n$', fc='#fde8e6')
    b_mad = box(2.55, 3.7, 2.0, 0.9, 'MAD Noise\nEstimation $\\hat{\\sigma}$', fc='#eaf2fb')
    b_wav = box(5.0, 3.7, 2.2, 0.9, 'Wavelet BayesShrink\nThresholding', fc='#eaf2fb')
    b_guide = box(7.65, 3.7, 2.0, 0.9, 'Wavelet\nEstimate $I_w$', fc='#eef7ed')

    b_bil = box(2.55, 1.9, 2.4, 0.9, 'Self-Guided\nBilateral Filter', fc='#eaf2fb')
    b_w = box(5.4, 1.9, 2.25, 0.9, 'Sobel Edge-Density\nWeight Map $w(x,y)$', fc='#eaf2fb')
    b_ib = box(8.1, 1.9, 1.9, 0.9, 'Bilateral\nOutput $I_b$', fc='#eef7ed')

    b_fuse = box(2.9, -0.15, 4.0, 0.9, 'Adaptive Fusion\n$I_{out} = w \\cdot I_b + (1-w)\\cdot I_w$', fc='#fdf2e3')

    arrow((b_in[0]+b_in[2], 4.15), (b_mad[0], 4.15))
    arrow((b_mad[0]+b_mad[2], 4.15), (b_wav[0], 4.15))
    arrow((b_wav[0]+b_wav[2], 4.15), (b_guide[0], 4.15))

    # noisy input feeds down into the self-guided bilateral filter directly
    arrow((b_in[0]+b_in[2]*0.5, b_in[1]), (b_bil[0]+0.3, b_bil[1]+b_bil[3]), connectionstyle='arc3,rad=0.18')
    # noise estimate also informs the bilateral filter's adaptive sigma_color
    arrow((b_mad[0]+b_mad[2]*0.5, b_mad[1]), (b_bil[0]+b_bil[2]-0.3, b_bil[1]+b_bil[3]), connectionstyle='arc3,rad=-0.15')
    # wavelet estimate feeds the weight map
    arrow((b_guide[0]+0.1, b_guide[1]), (b_w[0]+b_w[2]-0.1, b_w[1]+b_w[3]), connectionstyle='arc3,rad=-0.2')

    arrow((b_bil[0]+b_bil[2], 2.35), (b_w[0], 2.35))
    arrow((b_w[0]+b_w[2], 2.35), (b_ib[0], 2.35))

    arrow((b_bil[0]+b_bil[2]*0.5, b_bil[1]), (b_fuse[0]+0.4, b_fuse[1]+b_fuse[3]), connectionstyle='arc3,rad=0.12')
    arrow((b_ib[0]+b_ib[2]*0.5, b_ib[1]), (b_fuse[0]+b_fuse[2]-0.4, b_fuse[1]+b_fuse[3]), connectionstyle='arc3,rad=-0.12')
    arrow((b_guide[0]+b_guide[2]*0.5, b_guide[1]), (b_fuse[0]+b_fuse[2]-1.0, b_fuse[1]+b_fuse[3]), connectionstyle='arc3,rad=-0.3')

    ax.text(0.05, -0.28, 'Arrows: data flow.  Box colors -- input (pink), processing stage (blue), intermediate output (green), fusion (orange).', fontsize=8.0, color='#555555')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_pipeline_diagram.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig4_pipeline_diagram.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig4_pipeline_diagram")


# ----------------------------------------------------------------------
# Figure 5: average rank bar chart (n=240 conditions, both datasets)
# ----------------------------------------------------------------------
def fig_average_rank():
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
    ax.set_title('Mean Ranking Across 80 Images $\\times$ 3 Noise Levels (n=240)\n(BM3D excluded as a separate, non-lightweight complexity class)', fontsize=10)
    ax.grid(alpha=0.3, axis='x', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_average_rank.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig5_average_rank.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig5_average_rank")


# ----------------------------------------------------------------------
# Figure 6: visual comparison grid, on a BSD68 image (more representative
# than the original 3-image set), sigma=25, plus a zoomed inset
# ----------------------------------------------------------------------
def fig_visual_comparison():
    clean = np.array(Image.open(os.path.join(DATA_DIR, 'BSD68', 'test12.png')).convert('L'), dtype=np.float64)
    sigma = 25
    rng = np.random.RandomState(10000 * sigma + 0)
    noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255)
    sigma_est = estimate_noise_sigma(noisy)
    outputs = {
        'Noisy Input': noisy, 'Gaussian': baseline_gaussian(noisy), 'Median': baseline_median(noisy),
        'Bilateral': baseline_bilateral(noisy), 'NLM': baseline_nlm(noisy),
        'Wavelet-only': baseline_wavelet_only(noisy), 'BM3D': baseline_bm3d(noisy, sigma_est),
        'AWGB (Proposed)': awgb_denoise(noisy)[0], 'Ground Truth': clean,
    }
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.0))
    for ax, (name, img) in zip(axes.flat, outputs.items()):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(name, fontsize=11, fontweight='bold' if 'Proposed' in name else 'normal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1.2)
    plt.savefig(os.path.join(FIG_DIR, 'fig1_visual_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig1_visual_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig1_visual_comparison")


# ----------------------------------------------------------------------
# Figure 7: zoomed-crop error-map comparison (addresses reviewer request
# for "zoomed visual comparisons" and "error maps")
# ----------------------------------------------------------------------
def fig_zoom_errormap():
    clean = np.array(Image.open(os.path.join(DATA_DIR, 'BSD68', 'test12.png')).convert('L'), dtype=np.float64)
    sigma = 25
    rng = np.random.RandomState(10000 * sigma + 0)
    noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255)
    sigma_est = estimate_noise_sigma(noisy)
    wav_out = baseline_wavelet_only(noisy)
    awgb_out = awgb_denoise(noisy)[0]
    bm3d_out = baseline_bm3d(noisy, sigma_est)

    h, w = clean.shape
    y0, y1 = int(h*0.30), int(h*0.30)+100
    x0, x1 = int(w*0.40), int(w*0.40)+100
    crop = lambda im: im[y0:y1, x0:x1]

    methods_to_show = [('Ground Truth', clean), ('Wavelet-only', wav_out),
                        ('AWGB (Proposed)', awgb_out), ('BM3D', bm3d_out)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for col, (name, img) in enumerate(methods_to_show):
        axes[0, col].imshow(crop(img), cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(name, fontsize=10.5, fontweight='bold' if 'Proposed' in name else 'normal')
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        if name == 'Ground Truth':
            axes[1, col].imshow(np.zeros_like(crop(img)), cmap='inferno', vmin=0, vmax=40)
            axes[1, col].set_title('(reference)', fontsize=9, color='#777')
        else:
            err = np.abs(crop(img) - crop(clean))
            im = axes[1, col].imshow(err, cmap='inferno', vmin=0, vmax=40)
            axes[1, col].set_title(f'|error|, mean={err.mean():.2f}', fontsize=9)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
    axes[0, 0].set_ylabel('Zoomed crop', fontsize=10)
    axes[1, 0].set_ylabel('Absolute error', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_zoom_errormap.png'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig6_zoom_errormap.pdf'), bbox_inches='tight')
    plt.close()
    print("Saved fig6_zoom_errormap")


if __name__ == '__main__':
    fig_psnr_ssim_curves()
    fig_per_dataset_bars()
    fig_speed_quality_tradeoff()
    fig_pipeline_diagram()
    fig_average_rank()
    fig_visual_comparison()
    fig_zoom_errormap()
