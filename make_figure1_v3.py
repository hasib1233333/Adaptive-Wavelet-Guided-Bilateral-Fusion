"""
Comprehensive, high-resolution rebuild of Figure 1 (main visual comparison).

Improvements over the original:
- Higher-detail source image (BSD68 test34, portrait: fine texture + sharp
  edges + flat regions, so every method's behavior is visible)
- Each panel annotated with PSNR / SSIM computed against ground truth
- A zoomed inset box drawn directly on each panel, highlighting the same
  patch across every method for direct visual comparison
- Larger figure size and higher DPI for crisp print/zoom quality
- Cleaner typography and a consistent layout grid
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method import (
    awgb_denoise, baseline_gaussian, baseline_median, baseline_bilateral,
    baseline_nlm, baseline_wavelet_only, baseline_bm3d, estimate_noise_sigma
)

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'figure.dpi': 100,
    'savefig.dpi': 350,
})

FIG_DIR = '/home/claude/awgb/figures_v2'
DATA_DIR = '/home/claude/awgb/data'
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Load image and generate noisy version + all method outputs
# ----------------------------------------------------------------------
clean = np.array(Image.open(os.path.join(DATA_DIR, 'BSD68', 'test34.png')).convert('L'), dtype=np.float64)
sigma = 25
rng = np.random.RandomState(10000 * sigma + 0)
noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255)
sigma_est = estimate_noise_sigma(noisy)

print("Computing all method outputs...")
outputs = {
    'Noisy Input': (noisy, False),
    'Gaussian': (baseline_gaussian(noisy), True),
    'Median': (baseline_median(noisy), True),
    'Bilateral': (baseline_bilateral(noisy), True),
    'NLM': (baseline_nlm(noisy), True),
    'Wavelet-only': (baseline_wavelet_only(noisy), True),
    'BM3D': (baseline_bm3d(noisy, sigma_est), True),
    'AWGB (Proposed)': (awgb_denoise(noisy)[0], True),
    'Ground Truth': (clean, False),
}
print("Done.")

# Zoom region: chosen over the eye/forehead-wrinkle area for fine texture
# (wrinkles, eyebrow, eye detail) plus a nearby smooth-skin patch, so both
# edge-preservation and flat-region smoothing are visible in one crop.
h, w = clean.shape
zy0, zy1 = int(h * 0.40), int(h * 0.40) + 95
zx0, zx1 = int(w * 0.36), int(w * 0.36) + 95

# ----------------------------------------------------------------------
# Build the comprehensive 3x3 grid
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(14, 15.5))
order = ['Noisy Input', 'Gaussian', 'Median',
         'Bilateral', 'NLM', 'Wavelet-only',
         'BM3D', 'AWGB (Proposed)', 'Ground Truth']

for ax, name in zip(axes.flat, order):
    img, has_metric = outputs[name]
    img_disp = np.clip(img, 0, 255)
    ax.imshow(img_disp, cmap='gray', vmin=0, vmax=255, interpolation='nearest')

    # Title with method name
    is_proposed = 'Proposed' in name
    title_color = '#b3221c' if is_proposed else '#1a1a1a'
    ax.set_title(name, fontsize=15.5, fontweight='bold' if is_proposed else 'normal',
                 color=title_color, pad=8)

    # PSNR / SSIM annotation banner at bottom of image
    if has_metric:
        p = psnr_fn(clean, img_disp, data_range=255)
        s = ssim_fn(clean, img_disp, data_range=255)
        label = f'PSNR {p:.2f} dB   |   SSIM {s:.3f}'
        ax.text(0.5, 0.025, label, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=11.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.32', facecolor='#b3221c' if is_proposed else '#2c2c2c',
                          edgecolor='none', alpha=0.88))
    elif name == 'Noisy Input':
        p = psnr_fn(clean, img_disp, data_range=255)
        s = ssim_fn(clean, img_disp, data_range=255)
        label = f'PSNR {p:.2f} dB   |   SSIM {s:.3f}'
        ax.text(0.5, 0.025, label, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=11.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.32', facecolor='#555555', edgecolor='none', alpha=0.88))
    else:
        ax.text(0.5, 0.025, 'reference', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=11.5, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.32', facecolor='#555555', edgecolor='none', alpha=0.88))

    # Draw the zoom-region box on every panel (consistent location)
    rect = patches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                              linewidth=2.0, edgecolor='#ffd400', facecolor='none')
    ax.add_patch(rect)

    # Inset axes showing the zoomed crop, anchored to the corner
    axins = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
    axins.imshow(img_disp[zy0:zy1, zx0:zx1], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axins.set_xticks([]); axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor('#ffd400')
        spine.set_linewidth(2.0)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.suptitle(
    r'Denoising Comparison on BSD68 "test34" at Noise Level $\sigma=25$  (yellow box: zoomed inset region)',
    fontsize=17, y=0.997, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.975], pad=1.4, h_pad=2.2, w_pad=1.0)
plt.savefig(os.path.join(FIG_DIR, 'fig1_visual_comparison.png'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig1_visual_comparison.pdf'), bbox_inches='tight')
plt.close()
print("Saved comprehensive fig1_visual_comparison (PNG + PDF)")
