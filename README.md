# AWGB Paper Package -- v2 (Expanded, Post-Review)

**Adaptive Wavelet–Guided Bilateral Fusion: A Lightweight, Training-Free
Image Denoising Method for Resource-Constrained Visual Sensing Systems**

Md Hasibuzzaman, Dept. of Computer Science & Information Engineering,
Asia University, Taichung 41354, Taiwan.

This is a substantial revision of the original 3-image version, built in
response to a detailed external review. Read this file before submitting
anywhere -- it explains what changed and why, including one real bug that
was caught and fixed during the expansion.

## What changed since v1

| Reviewer concern | What we did |
|---|---|
| Too few images (3) | Expanded to the full **Set12 (12 images) + BSD68 (68 images) = 80 images**, the standard benchmark scale used throughout the denoising literature |
| No deep learning baselines | Added a **literature-cited comparison table** (DnCNN, FFDNet, DRUNet, SwinIR, Restormer) on BSD68 -- cited, not re-run, with reasoning for why re-running them ourselves would not have been a fair comparison |
| No statistical validation | Added **paired t-tests, Wilcoxon signed-rank tests, 95% CIs, and Cohen's d** for AWGB vs. every baseline, computed over all 240 (image, sigma) conditions |
| Limited references (17) | Expanded to **34 real, verifiable references**, including 2021-2025 work on lightweight/edge denoising |
| Figures not "journal-grade" | Added a **zoomed-crop + per-pixel error-map figure** and a per-dataset bar comparison; all figures are vector PDF, 300dpi PNG |
| Figure 1 too plain (follow-up request) | Rebuilt Figure 1 as a large, full-width comprehensive panel: real PSNR/SSIM stamped on every method, a consistent zoomed inset (forehead/eyebrow/eye) carried across all 9 panels on a higher-detail portrait image, so fine-detail differences are visible without a separate crop figure |
| Pipeline diagram too plain (follow-up request) | Rebuilt the architecture diagram (Fig. 1 in the paper) as a full-width figure with equation numbers on every block, clear three-stage grouping panels, and a real "what the image looks like at each stage" gallery showing actual pipeline outputs (not illustrative placeholders) at every step |
| "Preprint submitted to..." footer + date (follow-up request) | Removed via elsarticle's built-in `nopreprintline` class option -- no longer appears on page 1 |
| Gallery title overlapping first thumbnail label (follow-up request) | Fixed spacing in the pipeline diagram's right-hand gallery panel |
| Abstract over 250 words (follow-up request) | Trimmed to 249 words while keeping every essential claim: scope, headline result with real statistics, the ablation negative result, and the explicit non-superiority over BM3D/deep learning methods |
| "Limited scope: only grayscale images" (review concern) | Extended to **color images**: full Kodak24 (24 images) + McMaster (18 images) benchmarks, same 3-noise-level protocol, 126 further conditions. AWGB reproduces the grayscale result almost exactly (rank 1.48/6 vs. 1.29/6, +0.24dB vs wavelet-only vs +0.32dB, both p<10⁻²⁴) -- this is real, run, validated data, not a projection |
| Generative AI declaration required (journal/conference requirement) | Added a "Declaration of Generative AI and AI-assisted Technologies in the Writing Process" immediately above the references, per Elsevier's required placement. The statement accurately reflects that the author directed the research (scope, datasets, methodological decisions, the ablation study) and that Claude assisted with manuscript drafting, experimental code, and figure generation under that direction, with the author reviewing and verifying all output |
| **Citation verification (reviewer comment, [9],[10],[31],[32],[34])** | Manually verified every flagged reference against real search results. Found and fixed real problems: **two references were fully fabricated** (a non-existent IEEE Sensors Journal paper and a non-existent Sensors survey -- both removed and replaced with two already-verified real surveys); **two references had wrong author lists** (one credited "Xiong, Zhuoyuan" as first author when the real paper is by Zhuoqun Liu et al.; one credited "Liu, Jiale / Wu, Chenming / Wang, Yulun" for the Xformer paper when the real authors are Jiale Zhang, Yulun Zhang, Jinjin Gu, Jiahua Dong, Linghe Kong, Xiaokang Yang, published at ICLR 2024 not a generic arXiv preprint); **one reference had a wrong title/journal/volume/pages** (a Tian et al. survey was misattributed to a different title in Neurocomputing 2021, when the real paper is "Deep learning on image denoising: An overview" in Neural Networks 131 (2020) 251-275); **one reference was missing volume/pages** (the VRT paper, now corrected to vol. 33, pp. 2171-2182). All 32 final references are individually verified against real sources (not just pattern-matched) and 100% are cited in the text with no orphans either direction |
| Weak novelty | Not fully resolved (see "Honesty notes" below) -- we sharpened the framing to a systems/robustness contribution rather than claiming a new core algorithm |

## A real bug we found and fixed during the expansion

When we scaled from 3 images to 80, **AWGB's original design (a wavelet-*guided* joint
bilateral filter) actually lost to plain wavelet-only thresholding in 70% of
conditions.** This did not show up at the 3-image scale. We diagnosed it,
found the cause (the joint-bilateral output was too correlated with the
wavelet estimate to add independent information under fusion), fixed it
(switched to a **self-guided** bilateral filter on the noisy input instead),
and validated the fix on the full 80-image set: it now wins in 92% of
conditions. **This is reported as an ablation in Section 6 of the paper**,
not hidden -- we think it's a useful negative result for the field, and
removing it would mean overstating how clean the design process actually was.

If you adapt this paper for another use, please keep that section. It's
exactly the kind of rigor a reviewer wants to see, and it's real.

## What's in this package

```
AWGB_paper_v2.pdf      <- compiled paper (16 pages, Elsevier elsarticle format)
main.tex               <- LaTeX source
references.bib         <- 34 real, cited references
elsarticle.cls / .bst  <- Elsevier LaTeX class files (needed to compile)
figures/               <- 9 figures, PDF (vector) + PNG (300 dpi)
code/
  awgb_method.py             <- AWGB (corrected self-guided version) + all
                                 grayscale baselines, runnable
  awgb_method_color.py       <- per-channel RGB wrapper for AWGB + every
                                 baseline (color image extension)
  run_expanded_benchmark.py  <- full 80-image x 3-sigma grayscale benchmark
                                 (checkpointed, resumable)
  run_color_benchmark.py     <- full 42-image x 3-sigma color benchmark
                                 (checkpointed, resumable -- this is what
                                 actually ran for ~105 min)
  rerun_awgb_only.py          <- fast targeted rerun used after the bug fix
                                 (reuses cached baseline/BM3D results)
  make_figures_v2.py          <- regenerates the 5 core grayscale figures
  make_figure1_v3.py           <- regenerates the comprehensive Fig. 6
                                  (per-panel PSNR/SSIM + zoomed inset)
  make_figure4_v2.py           <- regenerates the pipeline diagram with
                                  real intermediate-stage thumbnails
  make_color_figures.py        <- regenerates the 2 color-image figures
results/
  expanded_results.json              <- FINAL grayscale results (80 images
                                         x 3 sigma x 7 methods)
  expanded_results_v1_OLD_buggy_fusion.json
                                      <- kept for transparency: the original
                                         (buggy) joint-bilateral AWGB results
  statistical_tests.json             <- grayscale paired t-test / Wilcoxon /
                                         CI / Cohen's d
  color_results.json                  <- FINAL color results (42 images x
                                         3 sigma x 7 methods)
  color_statistical_tests.json        <- color paired t-test / Wilcoxon /
                                         CI / Cohen's d
data/
  Set12/, BSD68/       <- 80 grayscale benchmark images
  Kodak24/, McMaster/  <- 42 color benchmark images
  (all fetched from the canonical cszn/FFDNet GitHub repository, the same
   source the original DnCNN/FFDNet papers' authors maintain)
```

## How to recompile the PDF

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## How to rerun the experiments from scratch

```
pip install numpy scipy opencv-python-headless PyWavelets scikit-image bm3d matplotlib pillow
python3 code/run_expanded_benchmark.py --max-seconds 280   # run in ~5 min chunks,
                                                             # checkpoints automatically;
                                                             # full run takes ~70 min
                                                             # (BM3D dominates: ~10s/image)
python3 code/make_figures_v2.py
```

## Honesty notes -- please read before submitting

0. **Two references in earlier drafts were fabricated, and this was only caught
   because you asked for verification.** When the reference list was first
   expanded (in response to the "add more references" review comment), two
   entries were invented rather than found: a non-existent IEEE Sensors
   Journal paper and a non-existent Sensors survey, both with plausible-
   sounding but fake author names. They have been removed. **This is exactly
   the failure mode academic publishers are watching for in AI-assisted
   writing, and it's a serious one** -- a fabricated citation in a submitted
   paper can be treated as research misconduct regardless of intent. If you
   add any further references to this paper yourself, or ask for more in a
   future session, verify each one independently (search for the exact
   title, confirm the author list and venue) before trusting it, rather than
   assuming any AI-generated reference list is correct by default.

1. **AWGB still does not beat BM3D, DnCNN, FFDNet, DRUNet, SwinIR, or
   Restormer.** It is the best *lightweight, training-free* method (rank
   1.29 of 6), with a statistically significant but numerically modest
   0.32 dB edge over wavelet-only thresholding alone. The abstract and
   conclusion say this plainly. Please don't strip that out -- it's the
   difference between a defensible claim and one a reviewer will catch
   in five minutes by reading Table 1.

2. **Novelty is still the weakest part of this paper**, and we did not
   try to manufacture a fix for that by inventing a new core algorithm.
   What we *did* do is sharpen the honest framing: this is a systems/
   robustness contribution (a fusion rule that avoids the worst-case
   failures of fixed-parameter classical methods), not a new denoising
   primitive. If you want to push novelty further, the most promising
   real lead in this package is the ablation finding itself (Section 6)
   -- the decorrelation principle it identifies could motivate a more
   original fusion design in a follow-up paper, rather than being
   retrofitted into this one.

3. **No journal has been selected.** `\journal{}` still says "Pending
   Journal Selection." An earlier search for a no-APC, IF 1-2 Elsevier
   journal did not find a clean match (most have drifted to IF 3-6); a
   broader search for legitimate non-Elsevier alternatives at IF 1-2 was
   started but not completed in this session.

4. **The deep-learning comparison table (Table 3) is literature-cited,
   not re-run.** This is disclosed in the text. If a reviewer wants a
   re-run comparison, that would require either officially released
   pretrained checkpoints matched to this exact noise protocol, or a
   full training pass -- both nontrivial additions, not something to
   fake.

5. **Three trials per condition (not five, as in v1).** This was a
   deliberate tradeoff to fit the much larger 80-image x 3-sigma grid
   (240 conditions, ~70 min of BM3D-dominated runtime) into the
   available session time. The statistical tests in Table 2 already
   account for this (n=240 paired comparisons, not n=240x3).

If you'd like, I can help next with: finishing the journal search,
drafting a cover letter, expanding to color images / Urban100/Kodak24,
or addressing any specific reviewer comments once you have them.
