# Round 2 Peer Review Report

**Paper Title:** FTMamba: Frequency-Aware Temporal Mamba for Long-Term Time Series Forecasting  
**Target Journal:** MDPI *Electronics* (ISSN 2079-9292, IF 2.6, CiteScore 6.1)  
**Review Date:** 2026-05-12  
**Reviewer:** Academic Peer Reviewer (Round 2 — venue-calibrated)

---

## 0. Calibration Note

This review is calibrated for MDPI *Electronics*, a mid-tier open-access journal covering computer science & engineering, AI, and signal processing. The acceptance threshold is appropriately lower than top-tier ML conferences (NeurIPS, ICML, ICLR). Contributions are expected to be sound, well-executed, and clearly presented, but do not need to be breakthrough-level novel. Transparent acknowledgment of limitations is valued. The review below reflects this calibration.

---

## 1. Overall Assessment

**Recommendation: Minor Revision (leaning toward Accept)**

The authors have made substantial improvements in response to Round 1 feedback. The revised manuscript is significantly more rigorous in its claims, more transparent about its limitations, and clearer in its technical descriptions. The paper now presents a credible, well-documented contribution that is appropriate for MDPI *Electronics*: a practical dual-branch architecture for time series forecasting with a non-obvious finding (gating > frequency features) and linear complexity. The remaining issues are minor and primarily concern MDPI-specific formatting and a few presentation refinements.

---

## 2. Assessment of Round 1 Revisions

### Major Issues — Resolution Status

| Issue | Round 1 Status | Round 2 Assessment |
|-------|---------------|-------------------|
| **M1** Asymmetric multi-seed | Required symmetric evaluation | **Partially resolved.** The Discussion now transparently identifies which horizons are within the seed-noise band (ETTh1/96, ETTh1/720) and which show clearer gains (ETTh2, ETTh1/336). The Limitations section explicitly flags the asymmetry. For an IF 2.6 journal, this level of transparency is acceptable. The paper no longer overclaims. |
| **M2** Missing FEDformer/FreTS | Required as baselines | **Acknowledged.** The Limitations section now states these baselines were omitted due to computational constraints and that their absence limits the strength of frequency-domain positioning claims. For this venue, this is sufficient — running these baselines would strengthen the paper but is not a hard requirement for acceptance given the paper's contributions. |
| **M3** Incomplete ablation | Required pure-Mamba variant, multi-horizon | **Acknowledged.** Limitations now explicitly lists the missing variant, the single-dataset/single-horizon scope, and what future work is needed. The existing two-variant ablation still supports the qualitative finding that gating matters more than the frequency branch, even if the quantification is approximate. |
| **M4** FFT-over-patches semantics | Critical | **Resolved.** The Frequency Branch section now opens with an explicit statement: "the FFT is applied over the patch dimension (index) of the learned embeddings, not over the raw time-series samples." The interpretation as a "learnable spectral mask" over patch-index frequencies is technically accurate and sets appropriate reader expectations. The gate dimension alignment (broadcast-expand [B,N,D] → [B,C,N,D]) is now documented. |
| **M5** Unverified FLOP numbers | Remove or measure | **Resolved.** Specific FLOP counts ("3.2M vs. 8.5M") have been removed. The asymptotic analysis is retained with appropriate caveats. The Limitations section notes the absence of empirical profiling. |
| **M6** Seed variance confounds claims | Qualify claims | **Resolved.** The Abstract, Discussion (§Training Variance), and Conclusions now use qualified language. The horizon-by-horizon breakdown of where FTMamba leads/trails in multi-seed mean is clear and honest. |

### Minor Issues — Resolution Status

All 10 minor issues (m1–m10) from Round 1 have been addressed with appropriate text changes. Notable improvements:

- **Abstract** is now ~190 words (within MDPI's ~200 word guideline) and free of specific percentages that invite overclaiming.
- **Figure 5 caption** now specifies "test sample index 42" and cross-references quantitative tables.
- **Batch size asymmetry** is explained (per-variate FFT memory cost on 21-channel Weather).
- **Code availability** statement added to Data Availability.
- **Gate parameterization** discussion added (per-element vs. scalar/channel/patch).
- **Weather dataset** now correctly cited to Rasp & Thuerey (ref1).
- **10-epoch training** is defended with convergence evidence.

### Overall Revision Quality: Good

The authors have been thorough and honest in their revisions. The paper is stronger for its transparency, and the qualified claims are more credible than the original's slightly overconfident framing.

---

## 3. MDPI Electronics — Venue-Specific Review

### 3.1 Fit with Journal Scope

*Electronics* covers computer science & engineering, AI, circuit and signal processing, and systems & control engineering. FTMamba sits at the intersection of **AI (deep learning architectures)** and **signal processing (FFT-based frequency analysis)**. The paper's emphasis on practical efficiency (linear complexity, modest parameter overhead) and its application to real-world forecasting problems (energy, weather) align with the journal's applied orientation.

**Assessment: Good fit.** The paper would benefit from briefly connecting its signal processing aspect (learnable complex-valued frequency filters) to the broader *Electronics* readership, who may be more familiar with filter design, spectral analysis, and DSP concepts than with the Mamba/SSM literature. A sentence or two in the Introduction bridging these communities would strengthen the fit.

### 3.2 MDPI Formatting Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| MDPI template (`electronics` class option) | ✓ | Correct document class |
| Abstract ≤ 200 words | ✓ | ~190 words after revision |
| Highlights section | ✓ | Present and correctly formatted |
| Numbered bracketed citations `[N]` | ✓ | Uses `\cite{refN}` with `thebibliography` |
| Full reference details (titles, DOIs) | △ | Refs 10–12 (Mamba, S-Mamba, TimeMachine) are listed as arXiv preprints. **Action needed:** Check whether these have since been published at peer-reviewed venues and update accordingly. If still arXiv-only, MDPI's editors may flag this. |
| Author contributions (CRediT) | ✓ | Present and detailed |
| Funding statement | ✓ | Present with grant numbers |
| Data availability statement | ✓ | Present (data public, code upon publication) |
| Conflicts of interest | ✓ | "The authors declare no conflicts of interest." |
| Institutional review / informed consent | ✓ | "Not applicable" |
| Abbreviations section | ✓ | Present |
| Publisher's note | △ | `\PublishersNote{}` is empty. For MDPI submissions, this is typically filled during production. Confirm with the MDPI template whether this should be left empty or include standard text. |
| Date fields (received/revised/accepted/published) | △ | All empty. Normal for initial submission; they will be filled by editorial office. |
| Graphical abstract | ? | Not included. MDPI encourages (but does not require) a graphical abstract. Consider adding one summarizing the dual-branch architecture visually. |
| Figure resolution (≥300 dpi) | ? | Cannot verify from LaTeX source. Ensure all PDF/PNG figures meet MDPI's resolution requirements before submission. |

### 3.3 MDPI Manuscript Structure

The current structure follows a conventional IMRaD pattern (Introduction → Related Work → Materials and Methods → Results → Discussion → Conclusions). MDPI has no strict section-name requirements, but the paper would read more naturally for this journal if:

- **Section 3** were titled "Methods" rather than "Materials and Methods" (no materials were used — this is purely computational).
- **Section 4** (Results) and **Section 5** (Discussion) could be merged, as MDPI articles often combine Results and Discussion. The current split is acceptable but makes Section 4.2 (Analysis) redundant with Section 5.

### 3.4 Reviewer Expectation Calibration

For *Electronics* (IF 2.6), reviewers typically expect:
- Sound methodology, clearly described ✓
- Adequate experiments on standard benchmarks ✓
- Honest acknowledgment of limitations ✓
- No requirement for SOTA on every benchmark △ (the paper claims 10/16 dataset-horizon pairs, which is good but not dominant)
- Reasonable novelty (incremental but well-executed is acceptable) ✓

**The paper meets the bar for this venue**, subject to the minor revisions below.

---

## 4. Remaining Issues (Round 2 — Minor)

### R2-1. Section Naming: "Materials and Methods" → "Methods"

The paper is a purely computational/algorithmic contribution. There are no physical materials, chemicals, or biological specimens. Renaming Section 3 from "Materials and Methods" to "Methods" is more accurate and aligns with MDPI conventions for computer science papers.

### R2-2. Reference Status Verification

References 10 (Mamba), 11 (S-Mamba), and 12 (TimeMachine) are listed as `arXiv` preprints. Before submitting to MDPI, verify:
- **Mamba (ref10):** First released on arXiv Dec 2023. Check whether a peer-reviewed version has been published (e.g., at ICML/NeurIPS 2024).
- **S-Mamba (ref11):** arXiv Mar 2024 — likely still a preprint. Consider updating if published in the interim.
- **TimeMachine (ref12):** arXiv Mar 2024 — same.

If any of these have been accepted at peer-reviewed venues, update the reference to the published version (with DOI). MDPI editors and iThenticate checks may flag excessive arXiv-only citations.

### R2-3. Figure File Resolution Verification

MDPI requires ≥300 dpi for photographs and ≥600 dpi for line art. The paper uses PDF figures, which are vector format — this is ideal. However, the associated PNG files (fig_ablation.png, fig_efficiency.png, fig_main_results.png, fig_multiseed.png, fig_prediction_curve.png) are also present in the figures directory. MDPI's production pipeline may use either format. Verify that all exported figures are generated at sufficient resolution. The `.png` files can be regenerated with `dpi=600` in matplotlib if needed.

### R2-4. Graphical Abstract Consideration

MDPI encourages graphical abstracts. Given this paper's strong visual element (dual-branch architecture diagram), a graphical abstract showing the FTMamba architecture with the key finding (gate > frequency branch) annotated would be an effective addition. This is optional but recommended to improve visibility and align with MDPI preferences.

### R2-5. Publisher's Note

The current `\PublishersNote{}` is empty. Check the latest MDPI LaTeX template documentation for the expected content. For initial submission, empty is typically acceptable (the publisher fills this during production). However, some templates now require a standard disclaimer sentence. Confirm before final submission.

### R2-6. Multi-Seed Figure Caption vs. Text Discrepancy

The caption of Figure 7 (fig_multiseed.pdf) states: "The low variance relative to inter-model gaps confirms that FTMamba's performance is architecturally grounded." However, the Discussion (§Training Variance) now acknowledges that at ETTh1/96 and ETTh1/720, the variance overlaps with the inter-model gap. The figure caption should be softened to match the more qualified claims in the text. Example revision: "The low variance relative to the typical inter-model gap suggests that FTMamba's performance is largely architectural, though at select horizons the seed-to-seed variance overlaps with inter-model differences (see Section 5.6)."

### R2-7. ETTm1 Spectral Dilution Argument

The Discussion (§5.4, L511) attributes ETTm1's weaker results to "spectral dilution" — the claim that 15-minute resolution spreads energy across a wider frequency band. This is a reasonable hypothesis but is stated as fact without quantitative support (no spectral analysis of ETTm1 vs. ETTh1 is shown). Consider adding a qualifier ("we hypothesize that...") or a brief spectral plot (e.g., dominant frequency analysis of ETTm1 vs. ETTh1) in the Appendix. Alternatively, move this claim to the Limitations section as an untested hypothesis.

### R2-8. Practical Application Discussion

For the *Electronics* audience (which includes practitioners and applied researchers), the paper would benefit from a brief paragraph in the Discussion or Conclusions about practical deployment considerations:
- Inference latency estimates (even if approximate)
- Memory footprint during inference
- Ease of implementation (the method reuses standard Mamba + FFT components, which is a practical advantage)
- Potential edge deployment scenarios (the linear complexity is relevant for resource-constrained settings)

This is not required for correctness but would improve the paper's relevance to the journal's readership.

---

## 5. Strengths (Post-Revision)

1. **Honest and transparent.** The qualified claims, expanded Limitations, and horizon-by-horizon breakdown of where results are/aren't significant set a good example for the field. This level of transparency is unusual and commendable.

2. **Clear technical exposition.** The FFT-over-patches clarification, the dimension alignment description in the gate, and the channel-independent design rationale make the architecture reproducible.

3. **Interesting finding.** The 4.0% vs. 0.9% ablation gap — that gating matters more than the presence of frequency features — is non-obvious and has implications beyond this specific architecture. It suggests that future multi-modal time series architectures should invest in fusion mechanisms, not just modality addition.

4. **Practical design.** The lightweight frequency branch (complex-valued filter bank, not a full MLP/attention module) and linear complexity make the method practically deployable, which aligns well with *Electronics*' applied readership.

5. **Comprehensive self-critique.** The Limitations section now covers 7 categories (frequency resolution, interpretability, benchmark diversity, baseline coverage, efficiency measurement, multi-seed evaluation, ablation scope, hyperparameter sensitivity) — more thorough than most published papers.

---

## 6. Summary & Recommendation

**Recommendation: Minor Revision**

The paper has been substantially improved through the Round 1 revision. The claims are now appropriately qualified, the technical ambiguities (FFT-over-patches, gate dimensions) are resolved, and the limitations are comprehensively documented. For MDPI *Electronics* (IF 2.6), the contribution — a practical dual-branch architecture with a non-obvious gating finding and linear complexity — meets the bar for publication.

The 8 remaining issues (R2-1 through R2-8) are all minor: section naming, reference verification, figure resolution, graphical abstract, publisher's note, figure caption alignment, spectral dilution caveat, and practical deployment discussion. None require new experiments.

### Priority Actions Before Submission:

| Priority | Issue | Effort |
|----------|-------|--------|
| **High** | R2-2: Verify and update arXiv references to published versions where available | 30 min |
| **High** | R2-6: Align multi-seed figure caption with qualified claims in text | 5 min |
| **Medium** | R2-1: Rename "Materials and Methods" → "Methods" | 2 min |
| **Medium** | R2-7: Add qualifier to spectral dilution argument | 5 min |
| **Medium** | R2-8: Add deployment considerations paragraph | 30 min |
| **Low** | R2-3: Verify figure resolution (regenerate PNGs at 600 dpi if needed) | 15 min |
| **Low** | R2-4: Consider graphical abstract | 1–2 hours |
| **Low** | R2-5: Confirm Publisher's Note format with latest MDPI template | 5 min |

---

## 7. Verdict Timeline

If the authors address the High and Medium priority items (R2-1, R2-2, R2-6, R2-7, R2-8 — approximately 1–2 hours of work), I expect the manuscript would be **acceptable for publication** in MDPI *Electronics* upon re-review. The core contribution is sound, the presentation is clear, and the limitations are transparently acknowledged.

---

*End of Round 2 Review*
