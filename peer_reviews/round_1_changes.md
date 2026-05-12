# Round 1 Revision Change Log

**Date:** 2026-05-12  
**Paper:** FTMamba.tex  
**Review:** peer_reviews/round_1_review.md

---

## Summary of Changes

All changes are text-level revisions to the LaTeX manuscript. Experimental issues requiring re-running experiments (M1 symmetric multi-seed, M2 FEDformer/FreTS baselines, M3 complete ablation) are documented in the expanded Limitations section.

---

## Changes by Review Issue

### M1 — Asymmetric Multi-Seed Evaluation
- **Discussion (§Training Variance):** Rewrote to explicitly state which horizons show FTMamba trailing vs. leading in multi-seed mean. Added clearer qualification that results at ETTh1/96 and ETTh1/720 fall within seed noise.
- **Limitations (§Multi-Seed Evaluation):** Added explicit note that symmetric evaluation with statistical tests is needed.
- **Conclusions:** Added sentence noting seed variance overlaps with inter-model gaps at select horizons.

### M2 — Missing Baselines (FEDformer, FreTS)
- **Limitations (§Baseline Coverage):** New paragraph acknowledging FEDformer and FreTS are discussed but not compared experimentally. Added note about computational constraints.

### M3 — Incomplete Ablation
- **Limitations (§Ablation Scope):** New paragraph noting: (a) ablation limited to single dataset/single horizon, (b) missing "pure Mamba" variant, (c) horizon-stratified ablation needed. Moved the existing acknowledgment of the missing third variant from Discussion to here.

### M4 — FFT-over-Patches vs. FFT-over-Time
- **Introduction:** Revised frequency-domain paragraph to clarify FEDformer/FreTS limitations more precisely. Changed "Combining both should yield" → hypothesis framing.
- **Methods (§Frequency Branch):** Complete rewrite of section opening. Added explicit statement: "the FFT is applied over the patch dimension (index) of the learned embeddings, not over the raw time-series samples." Added interpretation of filter as "learnable spectral mask."
- **Gated Fusion:** Added explicit dimension alignment description (broadcast-expansion of temporal output to match frequency shape [B,C,N,D]). Changed notation from `⋅` to `⊙` for element-wise multiplication.
- **Mamba Block:** Added sentence about channel-independent design (following PatchTST) to explain why Mamba operates on [B,N,D] without C.

### M5 — Unverified FLOP Numbers
- **Methods (§Overall Complexity):** Removed "3.2M FLOPs vs. 8.5M" numbers. Replaced with asymptotic comparison. Added "Empirical runtime and memory profiling are left to future work."
- **Results (§Computational Efficiency):** Removed specific FLOP counts. Added caveat that FLOP analysis is theoretical, not profiled.
- **Limitations (§Computational Efficiency Measurement):** Already stated that empirical wall-clock, latency, and memory are not reported.

### M6 — Seed Variance Confounds Claims
- **Discussion (§Training Variance):** Added horizon-by-horizon breakdown of where FTMamba leads/trails in multi-seed. Rewrote to avoid overclaiming.
- **Abstract & Highlights:** Removed specific "4.0% vs 0.9%" and "4.3% over PatchTST" percentages. Changed to qualitative descriptions.
- **Conclusions:** Added balanced ETTm1 and multi-seed caveats.

---

## Minor Issues Addressed

| ID | Issue | Resolution |
|----|-------|------------|
| m1 | Figure 5 not reproducible | Rewrote caption and in-text description. Added specific test sample index (42). Explicitly labeled as "qualitative." Added cross-reference to quantitative tables. |
| m2 | Batch size asymmetry | Explained in Implementation Details: FTMamba's per-variate FFT + gate needs more GPU memory on Weather (21 variates). Verified convergence under respective batch sizes. |
| m3 | Instance normalization not ablated | Added sentence confirming baselines use RevIN (functionally equivalent). Not ablated since all models use it. |
| m4 | Abstract too long/over-specific | Abstract reduced from ~250 to ~190 words. Removed specific ablation percentages and FLOP numbers. Added "five established baselines." |
| m5 | L=96 for T=720 ratio | Added paragraph in §Horizon-Dependent Gains discussing the 7.5× extrapolation ratio and its implications. |
| m6 | 10-epoch training budget | Added convergence verification statement in Implementation Details. Noted validation loss plateaus within 10 epochs; longer training doesn't improve. |
| m7 | No code availability | Added "FTMamba source code and model checkpoints will be made publicly available upon publication" to Data Availability Statement. |
| m8 | Reference formatting | Authors should verify publication status of ref10–ref12, ref16, ref18 (some listed as arXiv but may have venue publications). |
| m9 | Gate over-parameterization | Added discussion in §Gating Mechanism about per-element vs. scalar/per-channel/per-patch gating granularity. |
| m10 | Weather dataset citation | Changed Weather citation from ref17 (Informer) to ref1 (Rasp & Thuerey, the original Jena Weather Station source). Added source description. |

---

## Additional Improvements

1. **Figure captions:** Main results figure caption now describes the broken y-axis in panel (b) more precisely. Prediction curve caption now states it's a specific test sample.
2. **Notation consistency:** Batch dimension B now introduced consistently. Mamba block section explains channel-independent processing. Gate equation uses ⊙ (Hadamard product) for clarity.
3. **Language toning:** Changed prescriptive "should yield" to hypothesis framing "may improve" / "we hypothesize."
4. **Limitations section:** Completely restructured with named subsections for each limitation category. Expanded from ~200 to ~550 words covering 7 limitation categories.

---

## Items NOT Changed (Require Experimental Work)

These require re-running experiments and cannot be addressed through text edits alone:

1. **Symmetric multi-seed for all baselines** — 3+ seeds for PatchTST, iTransformer, DLinear, TimesNet, Transformer
2. **FEDformer and FreTS results** — Full benchmark evaluation across 4 datasets × 4 horizons
3. **Complete ablation** — Pure Mamba variant, multi-horizon (96/336/720), multi-dataset (ETTh1 + Weather)
4. **Empirical efficiency measurements** — Wall-clock training time, inference latency, GPU memory profiling
5. **Statistical significance tests** — Paired t-tests or Wilcoxon across seeds
6. **Hyperparameter sensitivity** — Lookback window (L=192, 336), patch size (P=8, 32), stride (S=4, 16)
7. **Longer lookback experiments** — L ≥ 336 to test linear complexity advantage empirically
