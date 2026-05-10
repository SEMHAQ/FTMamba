# Peer Review Report: FTMamba

**Paper:** FTMamba: Frequency-Aware Temporal Mamba for Long-Term Time Series Forecasting
**Target Journal:** MDPI Electronics
**Reviewer:** Automated Review (Claude Code)
**Date:** 2026-05-10

---

## Summary

This paper proposes FTMamba, a dual-branch architecture for long-term time series forecasting that combines Mamba-based temporal modeling with learnable frequency-domain filtering through a gated fusion mechanism. The authors evaluate on three ETT benchmark datasets (ETTh1, ETTh2, ETTm1) at horizons 96--720, comparing against five baselines. The paper is well-structured and follows MDPI formatting conventions. The core idea is sound and the results are generally positive, though several issues need attention before submission.

---

## Strengths

1. **Clear motivation.** The gap between Mamba-based temporal models and frequency-domain methods is well-articulated. The dual-branch design is a natural solution.

2. **Honest reporting.** The paper acknowledges that FTMamba does not win on all dataset--horizon pairs (e.g., ETTm1 at horizons 96--336 where PatchTST leads). This is preferable to cherry-picking.

3. **Ablation study.** The ablation isolates the contribution of the frequency branch and the gating mechanism separately, which is more informative than a single ablation.

4. **Complexity analysis.** The computational complexity comparison is clearly presented, with both theoretical Big-O and concrete FLOP estimates.

5. **Figure quality.** The five figures are publication-ready with consistent styling, proper broken-axis handling for ETTh2, and clear legends.

---

## Weaknesses

### W1: Limited Experimental Scope (Major)

Only three datasets are used, all from the ETT family. These share the same domain (electricity transformer temperature), the same number of variates (7), and similar periodicity structures. The paper would be significantly strengthened by adding at least one dataset from a different domain (e.g., Weather, Traffic, Electricity, or ILI). Without this, the generalizability claims are weak.

The authors acknowledge this in Section 6.7 (Limitations), but the limitation is serious enough to warrant either (a) adding experiments or (b) toning down the claims in the abstract and conclusions.

### W2: Multi-Seed Results Pending (Major)

Table 5 reports "pending" for all entries. The paper explicitly acknowledges a 1.3% discrepancy between the main results (MSE 0.3776) and the ablation (MSE 0.3826) on ETTh1/96, attributing it to different random seeds. This is the right thing to investigate, but the placeholder values must be filled before submission. A paper claiming superior performance cannot leave its reproducibility evidence incomplete.

### W3: No Statistical Significance Testing (Major)

Even when multi-seed results are available, the paper should report whether FTMamba's improvements over PatchTST are statistically significant. A paired t-test or Wilcoxon signed-rank test across seeds and horizons would strengthen the claims. Currently, a 1.3% improvement on ETTh1/96 (0.3776 vs 0.3827 for PatchTST) may not be significant given typical variance in deep learning experiments.

### W4: Prediction Curve is Simulated (Minor)

Figure 4 (prediction curve) uses simulated data (`np.random.seed(42)` with synthetic trend + daily + weekly components), not actual model outputs. The caption says "Forecast comparison on ETTh1" which implies real predictions. If this is a qualitative illustration, it should be labeled as such. If real predictions are available, they should be used instead.

### W5: Gating Mechanism Analysis is Thin (Minor)

The paper claims the gating mechanism is "the key component" (Section 6.2), but the analysis is purely quantitative. What does the gate actually learn? A brief visualization or analysis of gate values (e.g., does it prefer temporal at short horizons and frequency at long horizons?) would make this claim more convincing.

### W6: Only MSE/MAE Metrics (Minor)

The paper uses only MSE and MAE. Adding metrics like MASE (Mean Absolute Scaled Error) or SMAPE would provide a more complete picture, especially for comparing across datasets with different scales.

---

## Minor Issues

1. **Abstract length.** MDPI recommends abstracts of 150--250 words. The current abstract is approximately 130 words, which is on the short side. Consider adding one sentence about the key quantitative result.

2. **Figure 1 (architecture).** The paper references `figures/architecture.pdf` but this file was not found in the repository. Ensure it exists before submission.

3. **Reference formatting.** Reference 18 (Time-Series-Library) lacks a journal/proceedings entry. It is a GitHub repository. Consider citing the associated paper if one exists, or format it consistently with other references.

4. **ETTh2 bold values.** In Table 2, FTMamba is bold at horizons 96, 192, 336, but iTransformer is bold at 720. The text in Section 5.2 says FTMamba "leads on 3 of 4 horizons" for ETTh2, which is correct but could be more explicit about losing at T=720.

5. **Contribution list.** The four contributions in the Introduction (items 1--4) are somewhat formulaic. Consider consolidating to three stronger points.

6. **Typo check.** No significant typos found. The writing is clean and well-proofread.

7. **Backmatter order.** The MDPI template expects: authorcontributions, funding, institutionalreview, informedconsent, dataavailability, acknowledgments, conflictsofinterest. The current order is correct.

---

## Questions for Authors

1. How sensitive is FTMamba to the patch size P and stride S? The paper uses P=16, S=8 throughout. A brief sensitivity analysis would be helpful.

2. What is the parameter count of FTMamba compared to PatchTST and iTransformer? The FLOPs comparison is given, but parameter count affects memory and deployment.

3. Why was the lookback window fixed at L=96 for all experiments? Some baselines report results at L=96 and L=336. Does FTMamba's advantage hold at longer lookback windows?

4. The frequency branch uses a single FFT resolution. Have the authors tried multi-resolution FFT (different window sizes) or wavelet decomposition as mentioned in the limitations?

5. What happens when the frequency branch is removed but the gate is kept (i.e., gate between temporal and nothing)? This would clarify whether the gate's value comes from the frequency branch specifically or from the adaptive mixing mechanism in general.

---

## Formatting Issues (MDPI Compliance)

- [x] Document class: `[electronics,article,submit,pdftex,moreauthors]` -- correct
- [x] Backmatter order: correct
- [x] `\centering` in all figure environments -- correct
- [x] `\unskip` after figures -- correct
- [x] Abbreviations section present -- correct
- [x] `\PublishersNote{}` present -- correct
- [x] Funding statement filled -- correct
- [ ] Figure 1 (`architecture.pdf`) -- verify file exists
- [ ] Table 5 multi-seed results -- must be filled before submission

---

## Overall Recommendation

**Minor Revision.** The paper presents a solid contribution with a well-motivated dual-branch design. The main issues are: (1) limited dataset diversity, (2) pending multi-seed results, and (3) lack of statistical significance testing. Issues (2) and (3) must be addressed before submission. Issue (1) is a limitation that should be acknowledged but may not require additional experiments for a first submission. The minor issues and questions should also be addressed in the revision.

---

## Detailed Section-by-Section Review

### Introduction (Section 1)
- Well-structured, moves from problem to existing work to gap to contribution.
- The four contributions are clear but could be consolidated.
- No issues.

### Related Work (Section 2)
- Good coverage of Transformer-based, SSM-based, and frequency-domain methods.
- The transition to FTMamba at the end of Section 2.3 is smooth.
- Could mention more recent Mamba-based time series work (e.g., Vision Mamba, VMamba) if relevant.

### Materials and Methods (Section 3)
- The mathematical presentation is clear and well-formatted.
- The complexity analysis is thorough.
- Instance normalization is briefly mentioned; consider adding a sentence on why this is important for ETT datasets specifically.

### Results (Section 4)
- Tables are well-formatted with proper bolding.
- The broken-axis figure for ETTh2 is a good solution for the Transformer outlier values.
- The analysis subsection reads well and avoids overclaiming.

### Discussion (Section 5)
- The distinction between the frequency branch contribution (0.9%) and the gating contribution (4.0%) is the key insight and is well-articulated.
- The training variance discussion is honest and necessary.
- The limitations subsection is thorough.

### Conclusions (Section 6)
- Concise and accurate. Does not overclaim.
- Future work directions are specific and actionable.

---

*End of review.*
