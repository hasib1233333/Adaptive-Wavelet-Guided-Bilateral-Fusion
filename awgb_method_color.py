"""
Color-image extension of the AWGB benchmark.

All methods (AWGB and every baseline) are inherently grayscale operators.
Following standard practice in the denoising literature (e.g., DnCNN/FFDNet's
color variants, OpenCV's bilateralFilter and fastNlMeansDenoisingColored),
we apply each method independently to each of the three RGB channels and
recombine. This is a standard, simple extension -- not a novel contribution
-- and is described as such in the paper.
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method import (
    awgb_denoise, baseline_gaussian, baseline_median, baseline_bilateral,
    baseline_nlm, baseline_wavelet_only, baseline_bm3d, estimate_noise_sigma
)


def apply_per_channel(noisy_rgb, gray_fn, *args, **kwargs):
    """Apply a grayscale denoising function independently to each of the
    three channels of an (H, W, 3) float64 image in [0, 255]."""
    out = np.zeros_like(noisy_rgb)
    for c in range(3):
        out[:, :, c] = gray_fn(noisy_rgb[:, :, c], *args, **kwargs)
    return out


def awgb_denoise_color(noisy_rgb, wavelet='db4', level=3, d=9, win=7):
    out = np.zeros_like(noisy_rgb)
    sigma_per_channel = []
    for c in range(3):
        denoised, sigma_est, _ = awgb_denoise(noisy_rgb[:, :, c], wavelet, level, d, win)
        out[:, :, c] = denoised
        sigma_per_channel.append(sigma_est)
    return out, np.mean(sigma_per_channel)


def baseline_gaussian_color(noisy_rgb, ksize=5, sigma=1.2):
    return apply_per_channel(noisy_rgb, baseline_gaussian, ksize, sigma)


def baseline_median_color(noisy_rgb, ksize=5):
    return apply_per_channel(noisy_rgb, baseline_median, ksize)


def baseline_bilateral_color(noisy_rgb, d=9, sigma_color=40, sigma_space=5):
    return apply_per_channel(noisy_rgb, baseline_bilateral, d, sigma_color, sigma_space)


def baseline_nlm_color(noisy_rgb, h=12, templateWindowSize=7, searchWindowSize=21):
    return apply_per_channel(noisy_rgb, baseline_nlm, h, templateWindowSize, searchWindowSize)


def baseline_wavelet_only_color(noisy_rgb, wavelet='db4', level=3):
    return apply_per_channel(noisy_rgb, baseline_wavelet_only, wavelet, level)


def baseline_bm3d_color(noisy_rgb, sigma_noise):
    out = np.zeros_like(noisy_rgb)
    for c in range(3):
        out[:, :, c] = baseline_bm3d(noisy_rgb[:, :, c], sigma_noise)
    return out


def estimate_noise_sigma_color(noisy_rgb, wavelet='db4'):
    return np.mean([estimate_noise_sigma(noisy_rgb[:, :, c], wavelet) for c in range(3)])
