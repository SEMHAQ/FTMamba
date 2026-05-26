# Reviewer 3 (Round 1)

**Quality of English Language:** The English could be improved to more clearly express the research.

| Criteria | Yes | Can be improved | Must be improved | Not applicable |
|---|---|---|---|---|
| Does the introduction provide sufficient background and include all relevant references? |  | x |  |  |
| Is the research design appropriate? |  | x |  |  |
| Are the methods adequately described? |  | x |  |  |
| Are the results clearly presented? |  |  |  | x |
| Are the conclusions supported by the results? |  |  | x |  |
| Are all figures and tables clear and well-presented? |  | x |  |  |

## Comments and Suggestions for Authors

This paper proposes FTMamba, a dual-branch architecture combining Mamba (time-domain) with learnable FFT filters (frequency-domain) to address periodic pattern capture in long-term time series forecasting. Although the model achieves state-of-the-art results in 9 out of 16 experimental settings and demonstrates linear complexity advantages theoretically, the methodological justification, experimental rigor, and support for the core claims still have significant gaps. To meet publication standards, the authors must address the following issues through additional experiments and in-depth analysis.

**Major Revisions**

### 1. Insufficient validation of the core claim regarding gating mechanism effectiveness

The strong claim that "the gating mechanism is more important than the frequency branch itself" is currently supported only by incomplete ablation results on a single dataset and a single horizon.

(1) **Missing "pure Mamba" baseline.** The current ablation compares only "full model", "w/o frequency branch (gating retained)", and "simple addition (no gating)". In the "w/o frequency branch" variant, the gating is not actually removed but degenerates into a pass-through. A true "pure Mamba" baseline (removing both the frequency branch and gating) must be added.

(2) **Visualization of gating weights.** The claim that the gate "adaptively balances time and frequency domains based on input" lacks direct evidence. Extract gating weights on the test set and visualize their correlation with input characteristics such as periodicity and prediction horizon.

(3) **Comparison of different gating strategies.** Validate the necessity of element-wise gating by comparing at least three different granularities (global scalar gate, channel-wise gate, patch-wise gate).

### 2. Missing key frequency-domain and Mamba baselines

FEDformer, FreTS, S-Mamba, and TimeMachine are discussed in related work but not included as baselines. These comparisons are essential to justify the claimed methodological advantages.

### 3. Insufficient depth and breadth in the related work survey

Restructure related work from three dimensions:
- Classical architectures (GNN/RNN/CNN) — including GNNs for multivariate spatial dependencies (DOI: 10.1016/j.ymssp.2024.111841; DOI: 10.1109/JSEN.2024.3383665), RNNs for sequential modeling (DOI: 10.1007/978-981-99-1645-0_42), and recent CNN advances (DOI: 10.1016/j.neunet.2025.107139)
- Recent progress in Mamba-based architectures
- Frequency-domain forecasting models

### 4. Improvement of result visualization (Figure 4)

In Figure 4, prediction curves of different models largely overlap for most time steps. Add zoom-in insets of key prediction regions (peaks/troughs in the later part of the forecast) to highlight differences.

**Submission Date:** 18 May 2026
**Date of Review:** 21 May 2026 05:44:44
