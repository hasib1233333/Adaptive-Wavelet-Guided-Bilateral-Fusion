"""
Targeted rerun after the AWGB fusion-stage fix: reuse cached results for
every method whose code did not change (Gaussian, Median, Bilateral, NLM,
WaveletOnly, BM3D -- all unchanged), and recompute ONLY AWGB with the
corrected self-guided bilateral fusion design. This avoids re-running the
~70-minute BM3D pass for a fix that only touches AWGB's implementation.
"""
import os, sys, time, json, glob
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.dirname(__file__))
from awgb_method import awgb_denoise

DATA_DIR = '/home/claude/awgb/data'
OUT_DIR = '/home/claude/awgb/results'
OLD_PATH = os.path.join(OUT_DIR, 'expanded_results_v1_OLD_buggy_fusion.json')
NEW_PATH = os.path.join(OUT_DIR, 'expanded_results.json')

N_TRIALS = 3
SIGMAS = [15, 25, 50]


def load_dataset():
    images = {}
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'Set12', '*.png'))):
        name = 'Set12_' + os.path.basename(f).replace('.png', '')
        images[name] = np.array(Image.open(f).convert('L'), dtype=np.float64)
    for f in sorted(glob.glob(os.path.join(DATA_DIR, 'BSD68', '*.png'))):
        name = 'BSD68_' + os.path.basename(f).replace('.png', '')
        images[name] = np.array(Image.open(f).convert('L'), dtype=np.float64)
    return images


def add_gaussian_noise(img, sigma, seed):
    rng = np.random.RandomState(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)


def main():
    with open(OLD_PATH) as f:
        all_results = json.load(f)

    images = load_dataset()
    t0 = time.perf_counter()
    n = 0
    for img_name, clean in images.items():
        for sigma in SIGMAS:
            key = str(sigma)
            psnrs, ssims, times = [], [], []
            for trial in range(N_TRIALS):
                noisy = add_gaussian_noise(clean, sigma, seed=10000 * sigma + trial)
                t1 = time.perf_counter()
                out, _, _ = awgb_denoise(noisy)
                t2 = time.perf_counter()
                out = np.clip(out, 0, 255)
                psnrs.append(float(psnr_fn(clean, out, data_range=255)))
                ssims.append(float(ssim_fn(clean, out, data_range=255)))
                times.append((t2 - t1) * 1000.0)
            all_results[img_name][key]['AWGB (Proposed)'] = {
                'psnr_mean': float(np.mean(psnrs)),
                'psnr_std': float(np.std(psnrs)),
                'ssim_mean': float(np.mean(ssims)),
                'ssim_std': float(np.std(ssims)),
                'time_ms_mean': float(np.mean(times)),
                'psnr_trials': psnrs,
                'ssim_trials': ssims,
            }
            n += 1
        if n % 30 == 0:
            print(f"[{time.perf_counter()-t0:.1f}s] {n} conditions done", flush=True)

    with open(NEW_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Done. {n} conditions recomputed for AWGB. Saved to {NEW_PATH}")


if __name__ == '__main__':
    main()
