# Round 1 Peer Review Report

**Paper Title:** FTMamba: Frequency-Aware Temporal Mamba for Long-Term Time Series Forecasting  
**Review Date:** 2026-05-12  
**Reviewer:** Academic Peer Reviewer (Round 1)

---

## 1. Overall Assessment

**Recommendation: Major Revision**

FTMamba presents a dual-branch architecture that combines Mamba-based temporal modeling with learnable FFT-based frequency filtering for long-term time series forecasting. The core idea—that temporal and frequency information are complementary and should be adaptively fused via a learned gate—is intuitively appealing and reasonably well-motivated. The empirical results show competitive performance against established baselines on several benchmarks. However, the paper has significant weaknesses in experimental rigor, ablation design, novelty framing, and result interpretation that must be addressed before acceptance. The claimed contributions are modest relative to prior work (Mamba + FFT is a straightforward combination), and the experimental evidence does not fully support some of the paper's stronger claims.

---

## 2. Novelty & Contribution

**Assessment: Moderate**

**Strengths:**
- The dual-branch design with learnable complex-valued frequency filters and gated fusion is a reasonable architectural contribution, though the components are individually well-known.
- The finding that gated fusion matters more than the mere presence of frequency features (4.0% vs. 0.9% ablation gap) is interesting and worth reporting.

**Weaknesses:**
- The novelty is incremental. Mamba for time series has been explored (S-Mamba, TimeMachine). FFT with learnable filters in forecasting has been explored (FEDformer, FreTS). The main novelty is combining them with a gate, which is a standard attention/fusion mechanism (e.g., GLU, LSTM gates, SE-net, etc.). The paper needs to more precisely articulate what is *non-obvious* about this combination.
- The claim that FTMamba is the first to "co-learn temporal and frequency representations and merge them through a learned gate" (L97) needs verification against the broader gated / mixture-of-experts literature for time series. Are there really no prior works that fuse temporal and spectral features with learned weights?
- The frequency branch is essentially a complex-valued linear layer in the frequency domain (Equations 7–9). This is a lightweight add-on, not a substantial architectural innovation. The paper should be more upfront about the simplicity of this component.
- Contribution 2 ("10 of 16 dataset-horizon pairs") frames a mixed result as uniformly positive. On ETTm1, FTMamba leads on only 1 of 4 horizons. On ETTh2, it leads on 2 of 4. The framing should be more balanced.

---

## 3. Technical Soundness

**Assessment: Needs Improvement**

**Issues:**

1. **Gate Input Ambiguity (Critical).** Equation 10 concatenates `[h_temp; h_freq]` but the dimensions appear inconsistent. `h_temp` comes from the Mamba block operating on patches: expected shape `[B, N, D]` (batch, patches, model_dim). `h_freq` comes from IFFT operating on all channels: expected shape `[B, C, N, D]` (channel-dimension preserved through FFT). The paper does not explain how these are aligned before concatenation. Are channels averaged? Is there a projection? This is a crucial architectural detail.

2. **Ablation Design Flaw (Major).** The ablation has only two variants:
   - Full model → MSE 0.3826
   - w/o Frequency Branch (Mamba only) → MSE 0.3859 (+0.9%)
   - w/o Gated Fusion (simple addition) → MSE 0.3979 (+4.0%)

   **The missing variant is: w/o both gate AND frequency branch (pure Mamba + addition).** The paper acknowledges this in L499 but does not provide the number. Without it, we cannot distinguish whether the 4.0% degradation is from losing the gate, from losing the frequency features when they must be mixed via addition, or from the interaction of both. The interpretation in L425–426 is speculative without this variant.

   Additionally, "w/o Frequency Branch" where "the gate still operates but receives only temporal features" needs clarification: if the frequency output is set to zero, does Equation 11 collapse to `h_fused = g * h_temp + (1-g) * 0 = g * h_temp`? This would mean the gate acts as a *scalar* attenuation rather than a pass-through, which changes the interpretation entirely.

3. **FFT operates on patch sequences, not raw time series.** The input to FFT is the patch embedding, not the original signal. Since patch embedding is already a learned projection, the frequency domain here is over patch indices, not over the original time dimension. This fundamentally changes what "frequency" means—it's not seasonality or periodicity in the usual sense but rather the spectral content of learned patch representations. The paper glosses over this distinction. If patches are non-overlapping (stride = patch size), the FFT frequency axis corresponds to patch-index frequency, which has an indirect relationship to the original time-domain frequencies.

4. **Computational Complexity Analysis Issues.**
   - The FLOP comparison (L377: "3.2M FLOPs vs. 8.5M for PatchTST") is unsupported by any methodology or measurement. How were these numbers computed? Which FLOP counter was used?
   - The complexity analysis counts only the forward pass of the encoder but omits the patch embedding, prediction head, and FFT operations in the practical comparison.
   - The claim that PatchTST has "quadratic $O(N^2 \cdot D)$" cost (L506) is incorrect for PatchTST with $N=13$ patches—PatchTST's attention is over the patch dimension, which is already reduced from the original sequence length. The effective quadratic term is small. The paper should present actual wall-clock measurements.

5. **Single FFT Resolution Limitation.** The paper acknowledges (L522–523) that the frequency branch operates at a single resolution determined by patch count. This is a significant architectural limitation that should be discussed more prominently, as it means the model cannot capture multi-scale periodicities (e.g., daily + weekly + annual cycles simultaneously).

---

## 4. Experimental Quality

**Assessment: Below Standard**

1. **Asymmetric Multi-Seed Evaluation (Major).** PatchTST is evaluated with only a single seed while FTMamba uses 3 seeds. The paper acknowledges this limitation (L519, L529) but does not fix it. Given that seed-to-seed variance on ETTh1/96 (0.0034) is comparable to the claimed gain over PatchTST (0.3827 vs. 0.3776 = 0.0051), this is a serious concern. The multi-seed mean at ETTh1/720 is 0.5033 vs. PatchTST's single-seed 0.4878—the claimed gain disappears when accounting for variance. **Both models must be evaluated with the same number of seeds, and statistical significance tests are necessary.**

2. **Missing Baselines.** Given the paper's focus on frequency-domain modeling, the baselines should include:
   - **FEDformer** (cited as ref7, frequency-enhanced Transformer) — a direct frequency-domain competitor
   - **FreTS** (cited as ref16, frequency-domain MLP) — another direct frequency-domain competitor
   - **S-Mamba** or **TimeMachine** — Mamba-based time series methods that are cited as related work
   
   The absence of FEDformer and FreTS is particularly problematic because the paper positions itself against frequency-domain methods (Section 2.3, L95–97) but never compares against them experimentally.

3. **Ablation Study Scope.** The ablation is conducted on a single dataset (ETTh1) at a single horizon (T=96). The key claim—that gating matters more than frequency features—is supported by only one data point. Ablation results should be reported across multiple datasets and horizons to verify this finding generalizes.

4. **Limited Horizon Scope in Ablation.** The paper claims frequency features are more valuable at longer horizons (L511–513), but the ablation that isolates frequency contribution (removing the frequency branch) is only done at T=96. A horizon-stratified ablation (e.g., at T=96, 336, 720) is essential to support the claim.

5. **Figure 5 (Prediction Curve) is "Illustrative."** The caption describes it as "illustrative forecast comparison" showing "representative patterns observed during training." This is vague. Was this a specific test sample or a qualitative synthesis? If the latter, it should not be presented alongside quantitative results without clear disclaimers in the main text (not just the caption).

6. **No Hyperparameter Sensitivity.** The patch size (P=16), stride (S=8), number of layers (3), expand factor (2), and model dimension (512) are fixed. Sensitivity to these choices, particularly patch size (which determines the FFT resolution), is not explored.

7. **Training Details.** 10 epochs with Adam and learning rate 1e-4 is a very light training budget. Was convergence verified? Do baselines benefit from more training? Batch size differs dramatically between FTMamba (8 on Weather) and baselines (64 on Weather), which could advantage baselines that benefit from larger batches.

8. **Dataset Diversity.** Three of four benchmarks (ETTh1, ETTh2, ETTm1) are from the same source with the same features, differing only in station and sampling rate. The paper acknowledges this limitation (L525–526) but having only one truly independent dataset (Weather) weakens the generalizability claim.

---

## 5. Clarity & Presentation

**Assessment: Generally Clear, Some Issues**

1. **Architecture Diagram Missing Details.** The architecture figure is referenced (Fig. 1) but was not provided for review. Since the figure file exists at `figures/architecture.pdf`, I cannot verify whether it adequately conveys the dual-branch structure, dimension alignment, and gate mechanism.

2. **Notation Drift.** The problem formulation uses `C` for variates (L104), but in the architecture description, the patch embedding produces `[C, N, D]` (L120) while the Mamba block operates on `[B, N, D]` (L145). The batch dimension `B` appears mid-section without introduction. The channel dimension `C` is present in some equations but not others (e.g., Equation 3–4 lack `C`). This inconsistency makes the architecture hard to follow.

3. **Abstract is Too Long.** At roughly 250 words, the abstract exceeds the typical MDPI limit and contains excessive detail (specific ablation percentages, FLOP calculations). Streamline to core contributions and headline results.

4. **Line 367–375 (Analysis section)** reads more like discussion/conclusion than results analysis. Consider separating factual observations from interpretive claims.

5. **The discussion of seed-to-seed variance (L517–520)** contains self-critical analysis that is commendably transparent but undermines confidence in the main results. The paper needs to either:
   - Fix the issue by running symmetric multi-seed evaluations, OR
   - More carefully qualify the claims when seed variance overlaps with inter-model gaps.

6. **English and Grammar.** The writing quality is generally acceptable but has some awkward constructions. Examples:
   - L500: "this finding parallels the role of gating in other architectures" → overly broad claim about gating generality
   - L68: "Combining both should yield more reliable predictions" → prescriptive tone, should be "may yield" or "we hypothesize"

---

## 6. Major Issues

1. **M1 — Asymmetric Multi-Seed Evaluation.** FTMamba is evaluated with 3 seeds; PatchTST with 1. On ETTh1/720, FTMamba's multi-seed mean (0.5033) is worse than PatchTST's single-seed (0.4878), directly contradicting the claim of superiority. Run multi-seed evaluations for ALL baselines and report statistical significance tests (paired t-test or Wilcoxon).

2. **M2 — Missing Key Baselines.** FEDformer and FreTS are both frequency-domain methods cited and discussed in the paper but never included as experimental baselines. The paper's positioning against frequency-domain methods is unsubstantiated without these comparisons. At minimum, add FEDformer and FreTS results to all four benchmark tables.

3. **M3 — Incomplete Ablation.** The ablation misses the critical "pure Mamba + no gate" variant. The 4.0% vs. 0.9% gap is the paper's headline finding but is based on an incomplete experimental design. Add the variant and report across multiple datasets and horizons (at least ETTh1 + Weather at T=96, 336, 720).

4. **M4 — FFT-over-Patches vs. FFT-over-Time Ambiguity.** The frequency branch applies FFT to patch embeddings, not to the original time series. The paper repeatedly describes this as capturing "periodicity" and "seasonality" (L67, L362, L509–511), which is misleading. Clarify what "frequency" means in the context of patch-index FFT, and whether this actually captures the periodic structure the paper claims.

5. **M5 — Unverified Computational Efficiency Claims.** The FLOP comparisons are theoretical estimates without methodology. The paper claims linear complexity advantage but never reports wall-clock training time, inference latency, or GPU memory usage. Either provide measured benchmarks or remove unsupported numerical claims (e.g., "3.2M vs. 8.5M FLOPs").

6. **M6 — Seed Variance Confounds Main Claims.** Table 1 (ETTh1) reports FTMamba MSE 0.3776 at T=96; the ablation (Table 5) reports 0.3826 for the same configuration. The paper attributes this to different seeds (L517) but the 1.3% gap is comparable to the 1.3% gain over PatchTST at the same horizon. If seed variance equals the inter-model gap, the claimed improvements are not statistically meaningful. Fix with multi-seed reporting in main tables or qualify claims appropriately.

---

## 7. Minor Issues

1. **m1 — Figure 5 is not reproducible.** The "illustrative forecast comparison" should either be a specific test sample with index/identifier, or be moved to qualitative discussion and not presented as empirical evidence.

2. **m2 — Batch size asymmetry.** Weather: FTMamba batch=8, baselines=64. Explain why FTMamba cannot use larger batches (memory? convergence?) and whether this disadvantages baselines.

3. **m3 — Instance normalization is standard practice** (L219–232) but its contribution is never ablated. Given that DLinear and PatchTST also use RevIN/instance norm, it is unlikely to differentiate methods, but it should be stated whether baselines use the same normalization.

4. **m4 — The abstract's specific ablation percentages** (4.0%, 0.9%) belong in the results section, not the abstract. The abstract should summarize contributions qualitatively.

5. **m5 — Horizon 720 uses the same lookback (L=96)** as horizon 96. For a 720-step forecast from 96 historical steps, the model needs to extrapolate 7.5× beyond its input. Discuss whether this is reasonable and whether longer lookbacks would improve long-horizon results.

6. **m6 — Training epoch count (10 epochs)** is unusually low for deep time series models. Report learning curves or convergence evidence to show this is sufficient.

7. **m7 — No code or model checkpoint availability.** The acknowledgments mention using Time-Series-Library, but there is no statement about code release. Given the reproducibility concerns with seeds, code availability is important.

8. **m8 — Reference formatting.** Some references use arXiv IDs (ref10–ref12, ref16) while appearing to be published papers. Verify publication status. Ref16 (FreTS) is listed as NeurIPS 2024; this should be confirmed.

9. **m9 — The gate equation (Eq. 11): `h_fused = g * h_temp + (1-g) * h_freq`** uses element-wise multiplication. With `g` of shape `[B, C, N, D]`, this applies per-element gating which may be over-parameterized. Consider discussing whether a scalar or per-channel gate would be simpler and more interpretable.

10. **m10 — Section 4.1 (Datasets)** cites ref17 (Informer) as the source for all four datasets. Verify—the Weather dataset may originate from a different source.

---

## 8. Summary & Recommendation

FTMamba addresses a real problem (combining temporal and frequency information for time series forecasting) with a clean architectural design. The gated fusion finding—that *how* you combine matters more than *what* you combine—is potentially interesting. However, the paper in its current form has several issues that undermine confidence in the results:

1. **The central empirical claim is fragile.** Seed variance confounds the inter-model comparisons at key horizons.
2. **Critical baselines are missing** (FEDformer, FreTS), making the positioning against frequency-domain methods unsupported.
3. **The ablation is incomplete** and tested on only one data point.
4. **The technical description of the frequency branch** conflates FFT-over-patches with FFT-over-time, misleading readers about what "frequency" means in this architecture.
5. **Computational efficiency claims are unsubstantiated** by empirical measurements.

**I recommend Major Revision.** The paper is not fundamentally flawed and the core idea has merit, but the experimental evidence needs substantial strengthening before the claims can be considered reliable. The authors should either (a) conduct the additional experiments needed to support their claims, or (b) substantially qualify the claims to match the available evidence.

### Priority Revisions for Round 2:
1. Multi-seed evaluation for all baselines + statistical tests (M1, M6)
2. Add FEDformer and FreTS as baselines (M2)
3. Complete the ablation with the "pure Mamba" variant across multiple horizons (M3)
4. Clarify FFT-over-patches vs. FFT-over-time (M4)
5. Add empirical runtime/memory measurements (M5)

---

*End of Round 1 Review*
