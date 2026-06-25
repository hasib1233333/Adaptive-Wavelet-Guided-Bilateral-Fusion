"""
Color-image denoising benchmark: Kodak24 (24 images) + McMaster (18 images),
42 images total, 3 Gaussian noise levels, 3 trials each = 378 noisy images
processed per method. Checkpointed for resumable chunked execution.
"""
import os, sys, time, json, glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method_color import (
    awgb_denoise_color, baseline_gaussian_color, baseline_median_color,
    baseline_bilateral_color, baseline_nlm_color, baseline_wavelet_only_color,
    baseline_bm3d_color, estimate_noise_sigma_color
)

np.random.seed(42)

DATA_DIR = '/home/claude/awgb/data'
OUT_DIR = '/home/claude/awgb/results'
CKPT_PATH = os.path.join(OUT_DIR, 'color_results.json')
os.makedirs(OUT_DIR, exist_ok=True)

N_TRIALS = 3
SIGMAS = [15, 25, 50]


def load_dataset():
    images = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'Kodak24', '*.png'))):
        name = 'Kodak24_' + os.path.basename(f).replace('.png', '')
        img = np.array(Image.open(f).convert('RGB'), dtype=np.float64)
        images[name] = img
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'McMaster', '*.tif'))):
        name = 'McMaster_' + os.path.basename(f).replace('.tif', '')
        img = np.array(Image.open(f).convert('RGB'), dtype=np.float64)
        images[name] = img
    return images


def add_gaussian_noise(img, sigma, seed):
    rng = np.random.RandomState(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)


def run_all_methods(noisy, clean, include_bm3d=True):
    results = {}
    timings = {}

    def _eval(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = np.clip(fn(*args, **kwargs), 0, 255)
        t1 = time.perf_counter()
        results[name] = {
            'psnr': float(psnr_fn(clean, out, data_range=255)),
            'ssim': float(ssim_fn(clean, out, data_range=255, channel_axis=2)),
        }
        timings[name] = (t1 - t0) * 1000.0
        return out

    _eval('Gaussian', baseline_gaussian_color, noisy)
    _eval('Median', baseline_median_color, noisy)
    _eval('Bilateral', baseline_bilateral_color, noisy)
    _eval('NLM', baseline_nlm_color, noisy)
    _eval('WaveletOnly', baseline_wavelet_only_color, noisy)

    if include_bm3d:
        sigma_est = estimate_noise_sigma_color(noisy)
        _eval('BM3D', baseline_bm3d_color, noisy, sigma_est)

    def _awgb(img):
        out, _ = awgb_denoise_color(img)
        return out
    _eval('AWGB (Proposed)', _awgb, noisy)

    return results, timings


def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(data):
    tmp_path = CKPT_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, CKPT_PATH)


def main(max_seconds=None):
    t_start = time.perf_counter()
    images = load_dataset()
    all_results = load_checkpoint()
    n_trials_bm3d = 1  # BM3D color is ~47s/call; run once per condition

    n_done_this_run = 0
    for img_name, clean in images.items():
        if img_name not in all_results:
            all_results[img_name] = {}
        for sigma in SIGMAS:
            key = str(sigma)
            if key in all_results[img_name]:
                continue
            methods = ['Gaussian', 'Median', 'Bilateral', 'NLM', 'WaveletOnly', 'BM3D', 'AWGB (Proposed)']
            trial_psnr = {m: [] for m in methods}
            trial_ssim = {m: [] for m in methods}
            trial_time = {m: [] for m in methods}
            for trial in range(N_TRIALS):
                noisy = add_gaussian_noise(clean, sigma, seed=10000 * sigma + trial)
                include_bm3d = (trial < n_trials_bm3d)
                res, tim = run_all_methods(noisy, clean, include_bm3d=include_bm3d)
                for m, v in res.items():
                    trial_psnr[m].append(v['psnr'])
                    trial_ssim[m].append(v['ssim'])
                for m, t in tim.items():
                    trial_time[m].append(t)
            summary = {}
            for m in methods:
                summary[m] = {
                    'psnr_mean': float(np.mean(trial_psnr[m])),
                    'psnr_std': float(np.std(trial_psnr[m])),
                    'ssim_mean': float(np.mean(trial_ssim[m])),
                    'ssim_std': float(np.std(trial_ssim[m])),
                    'time_ms_mean': float(np.mean(trial_time[m])),
                }
            all_results[img_name][key] = summary
            save_checkpoint(all_results)
            n_done_this_run += 1
            elapsed = time.perf_counter() - t_start
            print(f"[{elapsed:6.1f}s] done: {img_name} sigma={sigma}", flush=True)
            if max_seconds and elapsed > max_seconds:
                print(f"Time budget {max_seconds}s reached, stopping chunk.", flush=True)
                return n_done_this_run
    return n_done_this_run


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-seconds', type=float, default=None)
    args = p.parse_args()
    n = main(max_seconds=args.max_seconds)
    print(f"Completed {n} (image, sigma) conditions this run.")
