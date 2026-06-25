"""
Comprehensive, visually rich rebuild of the AWGB pipeline diagram (Figure 4).

Improvements over the plain block diagram:
- Real image thumbnails embedded at every node, generated live from the
  actual pipeline running on a real noisy test image (not illustrative
  placeholders)
- Equation numbers/labels shown next to each processing block, matching
  the paper's numbered equations
- Clear visual grouping into "Wavelet branch" / "Bilateral branch" /
  "Fusion" with light background panels
- Larger, higher-resolution canvas with refined typography and spacing
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method import wavelet_denoise, guided_bilateral, edge_weight_map, estimate_noise_sigma

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'savefig.dpi': 350,
})

FIG_DIR = '/home/claude/awgb/figures_v2'
DATA_DIR = '/home/claude/awgb/data'
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Generate real intermediate outputs from the actual pipeline
# ----------------------------------------------------------------------
clean = np.array(Image.open(os.path.join(DATA_DIR, 'BSD68', 'test34.png')).convert('L'), dtype=np.float64)
sigma = 25
rng = np.random.RandomState(10000 * sigma + 0)
noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 255)
sigma_est = estimate_noise_sigma(noisy)
wav_out, _ = wavelet_denoise(noisy, sigma_noise=sigma_est)
bil_out = guided_bilateral(noisy, sigma_est)
w_map = edge_weight_map(wav_out)
fused = np.clip(w_map * bil_out + (1.0 - w_map) * wav_out, 0, 255)

# crop a tighter square-ish region (face only, not the full hat+head) so
# thumbnails stay compact and legible at small gallery size
h, w_ = clean.shape
cy0, cy1 = int(h * 0.18), int(h * 0.55)
cx0, cx1 = int(w_ * 0.05), int(w_ * 0.95)
thumb = lambda im: im[cy0:cy1, cx0:cx1]

print("Real pipeline outputs generated. sigma_est =", round(sigma_est, 2))


def add_thumb(ax, img, cx, cy, zoom_w, cmap='gray', vmin=0, vmax=255, border='#34495e', lw=1.4):
    """Place an image thumbnail centered at (cx, cy) in axes data coordinates,
    with width zoom_w (data units) and matching height by aspect ratio."""
    aspect = img.shape[0] / img.shape[1]
    zoom_h = zoom_w * aspect
    extent = [cx - zoom_w / 2, cx + zoom_w / 2, cy - zoom_h / 2, cy + zoom_h / 2]
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, zorder=4,
              interpolation='bilinear', aspect='auto')
    rect = patches.Rectangle((extent[0], extent[2]), zoom_w, zoom_h,
                              linewidth=lw, edgecolor=border, facecolor='none', zorder=5)
    ax.add_patch(rect)
    return zoom_h


# ----------------------------------------------------------------------
# Build the diagram
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 9.0))
ax.set_xlim(0, 13.6)
ax.set_ylim(-0.3, 9.2)
ax.axis('off')


def box(x, y, w, h, text, fc='#eef3f7', ec='#34495e', fontsize=10.3, lw=1.3, zorder=2):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                   facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, zorder=zorder + 1)
    return (x, y, w, h)


def arrow(p1, p2, lw=1.6, color='#34495e', connectionstyle=None, zorder=3):
    kwargs = dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=16)
    if connectionstyle:
        kwargs['connectionstyle'] = connectionstyle
    ax.annotate('', xy=p2, xytext=p1, arrowprops=kwargs, zorder=zorder)


# --- background grouping panels ---
panel_wav = patches.FancyBboxPatch((0.15, 5.55), 9.6, 2.85, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    facecolor='#fbfdfe', edgecolor='#b9cdd9', linewidth=1.1,
                                    linestyle='--', zorder=0.5)
ax.add_patch(panel_wav)
ax.text(0.35, 8.18, 'Wavelet Branch (Stage 1\u20132, Eq. 2\u20134)', fontsize=11.5, fontweight='bold',
        color='#3c5a73', zorder=1)

panel_bil = patches.FancyBboxPatch((0.15, 2.95), 9.6, 2.35, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    facecolor='#fffdfb', edgecolor='#d9c3a9', linewidth=1.1,
                                    linestyle='--', zorder=0.5)
ax.add_patch(panel_bil)
ax.text(0.35, 5.07, 'Self-Guided Bilateral Branch (Stage 3, Eq. 5\u20136)', fontsize=11.5, fontweight='bold',
        color='#8a5a2e', zorder=1)

panel_fuse = patches.FancyBboxPatch((0.15, 0.0), 9.6, 2.6, boxstyle='round,pad=0.02,rounding_size=0.12',
                                     facecolor='#fffaf3', edgecolor='#e0b97e', linewidth=1.1,
                                     linestyle='--', zorder=0.5)
ax.add_patch(panel_fuse)
ax.text(0.35, 2.32, 'Adaptive Fusion (Stage 4, Eq. 7\u20139)', fontsize=11.5, fontweight='bold',
        color='#a8631a', zorder=1)

# --- right-side thumbnail gallery panel ---
panel_gallery = patches.FancyBboxPatch((10.0, 0.0), 3.6, 9.0, boxstyle='round,pad=0.02,rounding_size=0.12',
                                        facecolor='#f7f7f7', edgecolor='#999999', linewidth=1.0, zorder=0.5)
ax.add_patch(panel_gallery)
ax.text(11.8, 8.92, 'What the image looks like\nat each stage', ha='center', va='top', fontsize=10.3, fontweight='bold',
        color='#333333', zorder=1)

# ---------------- WAVELET BRANCH ----------------
b_in = box(0.5, 6.35, 1.85, 0.85, 'Noisy Input\nImage $I_n$', fc='#fde3e0', fontsize=10.5)
b_mad = box(2.75, 6.35, 1.95, 0.85, 'MAD Noise\nEstimation $\\hat{\\sigma}$\n(Eq. 2)', fc='#dcebf7', fontsize=9.7)
b_wav = box(5.1, 6.35, 2.1, 0.85, 'Wavelet BayesShrink\nThresholding\n(Eq. 3\u20134)', fc='#dcebf7', fontsize=9.7)
b_guide = box(7.6, 6.35, 1.95, 0.85, 'Wavelet\nEstimate $I_w$', fc='#dceee0', fontsize=10.5)

arrow((b_in[0] + b_in[2], 6.775), (b_mad[0], 6.775))
arrow((b_mad[0] + b_mad[2], 6.775), (b_wav[0], 6.775))
arrow((b_wav[0] + b_wav[2], 6.775), (b_guide[0], 6.775))

# ---------------- BILATERAL BRANCH ----------------
b_bil = box(2.75, 3.85, 2.4, 0.85, 'Self-Guided\nBilateral Filter\n(Eq. 5\u20136)', fc='#dcebf7', fontsize=9.7)
b_w = box(5.55, 3.85, 2.2, 0.85, 'Sobel Edge-Density\nWeight Map $w(x,y)$\n(Eq. 7\u20138)', fc='#dcebf7', fontsize=9.2)
b_ib = box(8.15, 3.85, 1.5, 0.85, 'Bilateral\nOutput $I_b$', fc='#dceee0', fontsize=10.0)

# noisy input drops down into the bilateral filter (self-guided -> direct from I_n)
arrow((b_in[0] + b_in[2] * 0.5, b_in[1]), (b_bil[0] + 0.35, b_bil[1] + b_bil[3]), connectionstyle='arc3,rad=0.22')
# noise estimate informs the bilateral filter's adaptive sigma_c, sigma_s
arrow((b_mad[0] + b_mad[2] * 0.5, b_mad[1]), (b_bil[0] + b_bil[2] - 0.3, b_bil[1] + b_bil[3]),
      connectionstyle='arc3,rad=-0.18')
# wavelet estimate feeds the weight map (drives w from I_w's gradients)
arrow((b_guide[0] + 0.15, b_guide[1]), (b_w[0] + b_w[2] - 0.15, b_w[1] + b_w[3]), connectionstyle='arc3,rad=-0.22')

arrow((b_bil[0] + b_bil[2], 4.275), (b_w[0], 4.275))
arrow((b_w[0] + b_w[2], 4.275), (b_ib[0], 4.275))

# ---------------- FUSION ----------------
b_fuse = box(2.9, 0.55, 4.6, 1.15, 'Adaptive Fusion\n$I_{out}(p) = w(p)\\,I_b(p) + (1-w(p))\\,I_w(p)$\n(Eq. 9)',
             fc='#fcebd2', fontsize=10.8)

arrow((b_bil[0] + b_bil[2] * 0.5, b_bil[1]), (b_fuse[0] + 0.55, b_fuse[1] + b_fuse[3]),
      connectionstyle='arc3,rad=0.15')
arrow((b_ib[0] + b_ib[2] * 0.5, b_ib[1]), (b_fuse[0] + b_fuse[2] - 0.55, b_fuse[1] + b_fuse[3]),
      connectionstyle='arc3,rad=-0.15')
arrow((b_guide[0] + b_guide[2] * 0.5, b_guide[1]), (b_fuse[0] + b_fuse[2] - 1.5, b_fuse[1] + b_fuse[3]),
      connectionstyle='arc3,rad=-0.42')

# Final output thumbnail + label, to the right of the fusion box
b_out_label = box(7.9, 0.75, 1.75, 0.75, 'Denoised\nOutput $I_{out}$', fc='#fbe1c9', fontsize=10.0)
arrow((b_fuse[0] + b_fuse[2], b_fuse[1] + b_fuse[3] * 0.5), (b_out_label[0], b_out_label[1] + b_out_label[3] * 0.5))

# ---------------- thumbnail gallery (real outputs) ----------------
# Crop aspect ratio (height/width) for the thumbnail region:
_crop_h, _crop_w = thumb(noisy).shape
_aspect = _crop_h / _crop_w  # ~0.62 for this tighter face crop

gx = 11.8
thumb_w = 1.85  # data-units width; height = thumb_w * aspect
thumb_h = thumb_w * _aspect  # ~1.14

# 5 evenly spaced slot centers, gallery usable range from y=0.45 to y=7.65
# (leaving clearance below the title at y~8.0). Each slot reserves
# (thumb_h + label_gap) vertically; spacing chosen so consecutive labels
# never collide with the thumbnail above them.
top_y = 6.95
spacing = 1.58
slot_centers = [top_y - i * spacing for i in range(5)]
slot_labels = [
    f'Noisy input\n($\\hat\\sigma \\approx {sigma_est:.1f}$)',
    'Wavelet estimate $I_w$',
    'Bilateral output $I_b$',
    'Weight map $w(x,y)$',
    'Final output $I_{out}$',
]

for yc, label, img, cmap_, vmax_, border_, lw_ in zip(
    slot_centers, slot_labels,
    [thumb(noisy), thumb(wav_out), thumb(bil_out), thumb(w_map), thumb(fused)],
    ['gray', 'gray', 'gray', 'inferno', 'gray'],
    [255, 255, 255, 1, 255],
    ['#34495e', '#34495e', '#34495e', '#34495e', '#a8631a'],
    [1.3, 1.3, 1.3, 1.3, 2.2],
):
    label_y = yc + thumb_h / 2 + 0.16
    is_final = border_ == '#a8631a'
    ax.text(gx, label_y, label, ha='center', va='bottom', fontsize=9.2,
            color='#a8631a' if is_final else '#444', fontweight='bold' if is_final else 'normal')
    extent = [gx - thumb_w / 2, gx + thumb_w / 2, yc - thumb_h / 2, yc + thumb_h / 2]
    ax.imshow(img, cmap=cmap_, vmin=0, vmax=vmax_, extent=extent, zorder=4, aspect='auto')
    rect = patches.Rectangle((extent[0], extent[2]), thumb_w, thumb_h, linewidth=lw_,
                              edgecolor=border_, facecolor='none', zorder=5)
    ax.add_patch(rect)
    if yc != slot_centers[-1]:
        arrow((gx, yc - thumb_h / 2 - 0.03), (gx, yc - thumb_h / 2 - spacing + thumb_h / 2 + 0.20),
              lw=1.1, color='#999999', zorder=3)

ax.text(0.2, -0.22,
        'Arrows: data flow.  Dashed panels group the three processing stages.  Right column: actual pipeline outputs '
        'on a real noisy BSD68 image ($\\sigma=25$), shown for illustration.',
        fontsize=8.3, color='#555555')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_pipeline_diagram.png'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'fig4_pipeline_diagram.pdf'), bbox_inches='tight')
plt.close()
print("Saved comprehensive fig4_pipeline_diagram (PNG + PDF)")
