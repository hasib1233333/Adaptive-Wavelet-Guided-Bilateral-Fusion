"""
Expanded denoising benchmark: Set12 + BSD68, full BM3D on all images.
Checkpoints progress to disk after every image so long runs can be
resumed/chunked across multiple invocations without losing work.
"""
import os, sys, time, json, glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method import (
    awgb_denoise, baseline_gaussian, baseline_median, baseline_bilateral,
    baseline_nlm, baseline_wavelet_only, baseline_bm3d, estimate_noise_sigma
)

np.random.seed(42)

DATA_DIR = '/home/claude/awgb/data'
OUT_DIR = '/home/claude/awgb/results'
CKPT_PATH = os.path.join(OUT_DIR, 'expanded_results.json')
os.makedirs(OUT_DIR, exist_ok=True)

N_TRIALS = 3   # reduced from 5 (Section: original 3-image study) to fit
               # the much larger 80-image x 3-sigma grid in available time;
               # BM3D still runs on every trial (no longer subsampled).
SIGMAS = [15, 25, 50]


def load_dataset():
    images = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'Set12', '*.png'))):
        name = 'Set12_' + os.path.basename(f).replace('.png', '')
        img = np.array(Image.open(f).convert('L'), dtype=np.float64)
        images[name] = img
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'BSD68', '*.png'))):
        name = 'BSD68_' + os.path.basename(f).replace('.png', '')
        img = np.array(Image.open(f).convert('L'), dtype=np.float64)
        images[name] = img
    return images


def add_gaussian_noise(img, sigma, seed):
    rng = np.random.RandomState(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)


def run_all_methods(noisy, clean):
    results = {}
    timings = {}

    def _eval(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = np.clip(fn(*args, **kwargs), 0, 255)
        t1 = time.perf_counter()
        results[name] = {
            'psnr': float(psnr_fn(clean, out, data_range=255)),
            'ssim': float(ssim_fn(clean, out, data_range=255)),
        }
        timings[name] = (t1 - t0) * 1000.0
        return out

    _eval('Gaussian', baseline_gaussian, noisy)
    _eval('Median', baseline_median, noisy)
    _eval('Bilateral', baseline_bilateral, noisy)
    _eval('NLM', baseline_nlm, noisy)
    _eval('WaveletOnly', baseline_wavelet_only, noisy)
    sigma_est = estimate_noise_sigma(noisy)
    _eval('BM3D', baseline_bm3d, noisy, sigma_est)

    def _awgb(img):
        out, _, _ = awgb_denoise(img)
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


def main(max_seconds=None, image_filter=None):
    t_start = time.perf_counter()
    images = load_dataset()
    if image_filter:
        images = {k: v for k, v in images.items() if image_filter in k}
    all_results = load_checkpoint()

    n_done_this_run = 0
    for img_name, clean in images.items():
        if img_name not in all_results:
            all_results[img_name] = {}
        for sigma in SIGMAS:
            key = str(sigma)
            if key in all_results[img_name]:
                continue  # already computed in a previous chunk
            trial_psnr = {m: [] for m in ['Gaussian', 'Median', 'Bilateral',
                                           'NLM', 'WaveletOnly', 'BM3D', 'AWGB (Proposed)']}
            trial_ssim = {m: [] for m in trial_psnr}
            trial_time = {m: [] for m in trial_psnr}
            for trial in range(N_TRIALS):
                noisy = add_gaussian_noise(clean, sigma, seed=10000*sigma + trial)
                res, tim = run_all_methods(noisy, clean)
                for m, v in res.items():
                    trial_psnr[m].append(v['psnr'])
                    trial_ssim[m].append(v['ssim'])
                for m, t in tim.items():
                    trial_time[m].append(t)
            summary = {}
            for m in trial_psnr:
                summary[m] = {
                    'psnr_mean': float(np.mean(trial_psnr[m])),
                    'psnr_std': float(np.std(trial_psnr[m])),
                    'ssim_mean': float(np.mean(trial_ssim[m])),
                    'ssim_std': float(np.std(trial_ssim[m])),
                    'time_ms_mean': float(np.mean(trial_time[m])),
                    'psnr_trials': trial_psnr[m],   # keep raw trials for stats tests
                    'ssim_trials': trial_ssim[m],
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
    p.add_argument('--filter', type=str, default=None)
    args = p.parse_args()
    n = main(max_seconds=args.max_seconds, image_filter=args.filter)
    print(f"Completed {n} (image, sigma) conditions this run.")
