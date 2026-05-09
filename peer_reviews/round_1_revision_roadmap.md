# Revision Roadmap — Round 1

**Paper:** FTMamba
**Target:** MDPI Electronics
**Date:** 2026-05-09

---

## Priority 1: Critical (Must fix before submission)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Create architecture figure (figures/architecture.pdf) | Pending | Use draw.io or TikZ. Single column, ~8cm wide. See `figures/README.txt` for spec. |
| 2 | Add DOIs to all 19 references | Pending | Cross-ref each citation. MDPI requires DOIs. |
| 3 | Soften "state-of-the-art" claims | Pending | Abstract, Introduction item 4, Conclusions. Use "competitive" or "outperforms most baselines on ETTh1/ETTh2". |
| 4 | Add AI/GenAI disclosure | Pending | MDPI requires a statement. Add before References. |

## Priority 2: High (Strongly recommended)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5 | Run experiments with 3+ seeds | Pending | Report mean ± std for all results. Resolves MSE discrepancy. |
| 6 | Add S-Mamba and TimeMachine baselines | Pending | Both are Mamba-based, directly comparable. Code may need adaptation. |
| 7 | Add Weather or Traffic dataset | Pending | Weather has 21 variates, tests generalizability. |
| 8 | Tune Transformer baseline | Pending | Current MSE 1.9057 on ETTh2/96 is unreasonably high. |
| 9 | Increase training epochs | Pending | At least 20 epochs, or show convergence curves at 10. |
| 10 | Extend ablation to T=720 | Pending | Supports the claim that frequency branch helps at longer horizons. |

## Priority 3: Medium (Should fix)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 11 | Trim abstract to ≤200 words | Pending | Currently ~220 words. |
| 12 | Add hyperparameter sensitivity | Pending | Patch length P, number of layers N_l at minimum. |
| 13 | Visualize learned frequency filters | Pending | Plot F_real after training. Shows which frequencies are emphasized. |
| 14 | Fix residual AI patterns in Analysis | Pending | "demonstrates strong performance", "validates the effectiveness" etc. |
| 15 | Address DA challenges in Discussion | Pending | Especially DA-C1 (is frequency branch necessary?) and DA-C3 (SOTA claim). |

## Priority 4: Low (Nice to have)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 16 | Wall-clock time comparison | Pending | FLOPs don't always translate to speed. |
| 17 | Per-variate frequency filter | Pending | Addresses R1-W1. |
| 18 | Longer lookback window (L=336, 512) | Pending | Addresses R1-W2 (frequency resolution). |
| 19 | Cross-attention gating ablation | Pending | Addresses DA-C5. |

---

## Estimated Effort

- Priority 1: ~1 day (figure is the bottleneck)
- Priority 2: ~2-3 days (experiments need GPU time)
- Priority 3: ~0.5 day (mostly text edits)
- Priority 4: Optional, ~1-2 days if pursued

**Total estimated time to submission-ready: 4-6 days**

---

*Next step: After completing Priority 1-2 items, run Round 2 review.*
