# Peer Review Report — Round 1

**Paper:** FTMamba: Frequency-Aware Temporal Mamba for Long-Term Time Series Forecasting
**Target Journal:** MDPI Electronics (CAS Q4)
**Date:** 2026-05-09
**Review Panel:** EIC + 3 Peer Reviewers + Devil's Advocate

---

## 1. Editor-in-Chief (EIC) — Overall Assessment

**Recommendation: Minor Revision**

The paper presents FTMamba, a dual-branch architecture combining Mamba's temporal modeling with learnable frequency-domain filtering for long-term time series forecasting. The topic is timely and the approach is reasonable. The writing has been improved (recent humanization pass removed most AI-typical patterns). However, several issues must be addressed before acceptance.

### Critical Issues (Must Fix)

| # | Issue | Severity |
|---|-------|----------|
| E1 | **Missing architecture figure** — `figures/architecture.pdf` is referenced but does not exist. The paper cannot compile without it. | Critical |
| E2 | **Ablation MSE discrepancy** — Full model reports 0.3776 (Table 2) vs 0.3826 (Table 5) on ETTh1/96. The Discussion now acknowledges this, but it should be resolved: either re-run with the same seed or report mean ± std. | High |
| E3 | **Only 3 datasets** — ETTh1, ETTh2, ETTm1 are all from the ETT family. Weather, Traffic, or Electricity should be added for generalizability. | High |
| E4 | **Missing DOIs** — None of the 19 references include DOI numbers. MDPI requires DOIs for all citable items. | High |
| E5 | **10 epochs is very low** — Standard practice is 20-100 epochs. With only 10 epochs and no convergence curves, reviewers may question whether results are under-trained. | Medium |

### Minor Issues

| # | Issue |
|---|-------|
| E6 | Abstract is ~220 words; MDPI recommends ≤200. Trim by ~20 words. |
| E7 | `\datereceived`, `\daterevised`, `\dateaccepted`, `\datepublished` are all blank — fill before submission. |
| E8 | No AI/GenAI disclosure statement. MDPI now requires one. |
| E9 | `funding` says "no external funding" — verify this is accurate. |
| E10 | Reference 18 (Time-Series-Library) uses "Available online" format — should be formalized with a proper citation. |

---

## 2. Reviewer 1 (R1) — Methodology Review

**Confidence:** 4/5
**Recommendation: Accept with Minor Revisions**

### Strengths
- Clear problem formulation and well-motivated architecture design
- The dual-branch concept (temporal + frequency) is intuitive and well-justified
- Computational complexity analysis is thorough and correctly derived
- Gated fusion mechanism is a sensible design choice

### Weaknesses

**R1-W1: Learnable frequency filter is under-specified (Medium)**
The filter $\mathbf{F} \in \mathbb{C}^{1 \times 1 \times (N/2+1)}$ is shared across all batch and channel dimensions. This means every variate and every sample uses the same frequency weighting. Have the authors considered per-channel or per-variate filters? The current design may under-fit for multivariate data with heterogeneous periodic structures.

**R1-W2: Patch count is very small (Medium)**
With $L=96$, $P=16$, $S=8$, the number of patches is $N = \lfloor(96-16)/8\rfloor + 2 = 12$. The FFT of 12 points yields only 7 frequency bins. This severely limits the frequency resolution. The authors should discuss this limitation and ideally test with longer lookback windows (e.g., $L=336$ or $L=512$).

**R1-W3: No hyperparameter sensitivity analysis (Low)**
The paper fixes all hyperparameters without ablation on key choices: patch length $P$, stride $S$, number of layers $N_l$, expansion factor. A sensitivity analysis on at least $P$ and $N_l$ would strengthen the claims.

**R1-W4: Instance normalization placement is unclear (Low)**
Section 3.6 describes instance normalization but it's not shown in the architecture diagram description or equations. Where exactly is it applied — before patch embedding, after the prediction head, or both? Clarify in the figure caption or add an equation showing the full pipeline with normalization.

### Questions
1. What is the learned frequency filter actually weighting? Can you visualize $\mathbf{F}_{\text{real}}$ after training to show which frequencies are emphasized?
2. How does performance change with $L=336$ or $L=512$?

---

## 3. Reviewer 2 (R2) — Experimental Review

**Confidence:** 4/5
**Recommendation: Major Revision**

### Strengths
- Results are presented clearly with MSE/MAE metrics
- Ablation study isolates the contribution of each component
- The paper reports improvements on 3/4 horizons for ETTh1 and ETTh2

### Weaknesses

**R2-W1: Insufficient baselines (High)**
Only 5 baselines are compared. The paper omits several strong recent methods:
- **Mamba-based:** S-Mamba, TimeMachine (both cited but not compared)
- **Frequency-based:** FreTS (cited but not compared)
- **Linear:** DLinear is included but RLinear or FITS are missing
- **Other SOTA:** TimesFM, Chronos, or other foundation models

For MDPI Electronics, at least 8-10 baselines are expected. The current selection may appear cherry-picked.

**R2-W2: No statistical significance (High)**
All results are single-run. With the acknowledged 1.3% variance between runs (Discussion), it's unclear whether FTMamba's 1.3% improvement over PatchTST on ETTh1/96 is statistically significant. Report mean ± std over 3-5 runs with different seeds.

**R2-W3: ETT-only evaluation is weak (High)**
Three datasets from the same family (ETT) is insufficient. The Weather dataset (21 variates, strong seasonality) and Traffic dataset (862 variates) are standard benchmarks. Not including them weakens the generalizability claim.

**R2-W4: Transformer baseline appears poorly tuned (Medium)**
The vanilla Transformer shows extremely poor results (e.g., MSE 1.9057 on ETTh2/96 vs 0.2904 for FTMamba). This is suspicious — a well-tuned Transformer should achieve ~0.35-0.40 on ETTh2/96. Were the Transformer hyperparameters optimized? If not, the comparison is unfair.

**R2-W5: No training curves (Low)**
With only 10 epochs, showing MSE vs. epoch curves for FTMamba and baselines would help verify convergence. Did the models converge, or would more training help?

**R2-W6: Computational efficiency claims lack wall-clock evidence (Low)**
The paper claims 3.2M vs 8.5M FLOPs, but FLOPs don't always translate to wall-clock time. Report actual training time and inference latency.

### Questions
1. Why were S-Mamba and TimeMachine not included as baselines, given they are the most directly comparable Mamba-based methods?
2. Can you provide training curves for 10 epochs to show convergence?

---

## 4. Reviewer 3 (R3) — Writing and Presentation Review

**Confidence:** 5/5
**Recommendation: Accept with Minor Revisions**

### Strengths
- The writing is clear and direct (recent humanization improved readability significantly)
- Mathematical notation is consistent and well-defined
- The paper follows a logical structure: motivation → method → results → discussion
- The Discussion section honestly addresses limitations (MSE discrepancy, dataset-specific performance)

### Weaknesses

**R3-W1: Abstract still uses "state-of-the-art" (Medium)**
The abstract claims "state-of-the-art performance" but PatchTST beats FTMamba on ETTm1 at horizons 96, 192, 336, and on ETTh2 at horizon 192. Soften to "competitive performance" or specify "on ETTh1 and ETTh2."

**R3-W2: Analysis section still has residual AI patterns (Low)**
Section 4.2 (Analysis) still contains some formulaic phrases:
- "demonstrates strong performance" → "performs well"
- "validates the effectiveness of the Mamba-based architecture" → remove or be specific
- "The substantial improvements over DLinear highlight the importance of..." → simplify

**R3-W3: Contribution list item 4 is weak (Low)**
"Experiments on three benchmarks" is a contribution of work, not a contribution of knowledge. Reframe as: "We show that frequency-domain features are most beneficial at longer horizons, with gains increasing from 1.3% at T=96 to 4.3% at T=720 on ETTh1."

**R3-W4: Missing limitation on single-FFT resolution (Low)**
The Discussion mentions this, but it should also appear in the Conclusions as an explicit limitation.

**R3-W5: Table 5 ablation is incomplete (Medium)**
The ablation only tests on ETTh1/96. Testing on at least one more horizon (e.g., 720) would show whether the frequency branch becomes more important at longer horizons — which would directly support the paper's central claim.

### Minor Issues
- Line 222: "crucial" still appears once in Section 3.6 (Instance Normalization) — replace with "important" or "necessary"
- Reference formatting: some refs use "In Proceedings of..." while others use journal format — standardize
- The Acknowledgments section thanks "anonymous reviewers" — this is template boilerplate; personalize or remove

---

## 5. Devil's Advocate (DA) — Challenge Review

**Purpose:** Stress-test the paper's core claims and identify the strongest counter-arguments.

### DA-C1: Is the frequency branch actually necessary?
The ablation shows only 0.9% MSE loss when removing the entire frequency branch. The gated fusion adds 4.0%, but this could be because the gate acts as a regularizer or attention mechanism, not because frequency features are valuable. Have you tested a gated fusion of two temporal branches (e.g., two Mamba blocks with different configurations)? If that matches the full model, the frequency domain is irrelevant.

### DA-C2: The 10-epoch training protocol may invalidate all comparisons
10 epochs is far below standard practice. If baselines were also trained for only 10 epochs, they may be under-trained relative to FTMamba (which may converge faster due to the simpler architecture). This creates a systematic bias in FTMamba's favor. The fair comparison requires either (a) training all models to convergence with early stopping, or (b) showing convergence curves.

### DA-C3: The "state-of-the-art" claim is not supported
FTMamba loses to PatchTST on:
- ETTh1/336 (0.4931 vs 0.4857)
- ETTh2/192 (0.3784 vs 0.3757)
- ETTm1/96, 192, 336 (all three horizons)

That's 5 out of 12 dataset-horizon pairs where FTMamba is not the best. "Outperforms on most" is defensible; "state-of-the-art" is not.

### DA-C4: The architecture is a straightforward composition
Mamba + FFT + gated fusion is not a novel architecture — it's a direct combination of existing components. The novelty claim should be modest: "we show that combining existing temporal and frequency components with a learned gate is effective" rather than "we design a novel architecture."

### DA-C5: Missing ablation of the gating mechanism design
The paper uses $\mathbf{g} = \sigma(\text{Linear}([\mathbf{h}_{\text{temp}}; \mathbf{h}_{\text{freq}}]))$. Why not other designs: dot-product attention, cross-attention, or simply averaging? The gate design itself is not ablated.

---

## Summary of Required Revisions

### Must Fix (Before Resubmission)
1. Create `figures/architecture.pdf` or remove the figure reference
2. Add 2-3 more datasets (Weather, Traffic, or Electricity)
3. Add DOIs to all references
4. Run experiments with 3+ seeds and report mean ± std
5. Add S-Mamba and TimeMachine as baselines
6. Soften "state-of-the-art" claims throughout
7. Add AI/GenAI disclosure statement
8. Tune Transformer baseline or explain the poor performance

### Should Fix (Strongly Recommended)
9. Extend ablation to more horizons (at least T=720)
10. Increase training epochs to 20+ or show convergence curves
11. Add hyperparameter sensitivity analysis (patch length, number of layers)
12. Trim abstract to ≤200 words
13. Visualize learned frequency filters

### Nice to Have
14. Wall-clock training/inference time comparison
15. Per-variate frequency filter variant
16. Cross-attention gating ablation
17. Longer lookback window experiments (L=336, 512)

---

*Review generated by simulated multi-perspective peer review. For revision guidance, see `round_1_revision_roadmap.md`.*
