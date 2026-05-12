# Round 3 Peer Review Report

**Paper Title:** FTMamba: Frequency-Aware Temporal Mamba for Long-Term Time Series Forecasting  
**Target Journal:** MDPI *Electronics* (IF 2.6)  
**Review Date:** 2026-05-12  
**Review Stage:** Final polish review (post-R1 + R2 revisions)

---

## Overall Assessment

**Recommendation: Accept with Minor Corrections (essentially ready)**

The manuscript has been substantially improved through two rounds of revision. The claims are now appropriately qualified, the technical description is clear and reproducible, and the limitations are honestly documented. The paper is well-calibrated for MDPI *Electronics*. This round identifies only a handful of remaining issues — one counting error that must be fixed, and a few presentation inconsistencies.

---

## R3-1 (Critical): "10 of 16" → "9 of 16" — Counting Error

The paper consistently states that FTMamba achieves best MSE on **10 of 16** dataset–horizon pairs. A direct count from Tables 1–4 (based on bolded MSE values) yields **9 of 16**:

| Dataset | 96 | 192 | 336 | 720 | FTMamba Wins |
|---------|----|-----|-----|-----|-------------|
| ETTh1 | FTMamba ✓ | FTMamba ✓ | PatchTST | FTMamba ✓ | 3/4 |
| ETTh2 | FTMamba ✓ | PatchTST | FTMamba ✓ | iTransformer | 2/4 |
| ETTm1 | PatchTST | PatchTST | PatchTST | FTMamba ✓ | 1/4 |
| Weather | PatchTST | FTMamba ✓ | FTMamba ✓ | FTMamba ✓ | 3/4 |
| **Total** | | | | | **9/16** |

**Locations requiring correction:**
- L36 (Abstract): "on 10 of 16 dataset--horizon pairs"
- L45 (Highlights): "10 of 16 dataset--horizon pairs"
- L74 (Contributions): "10 of 16 dataset--horizon pairs"
- L548 (Conclusions): "10 of 16 dataset--horizon pairs"

Recommended replacement: "on 9 of 16 dataset--horizon pairs" in all four locations.

Alternatively, if the authors are counting by a different criterion (e.g., counting both MSE and MAE wins, or counting "best or second-best"), the criterion must be stated. But "best MSE" is the natural interpretation given the table bolding convention.

---

## R3-2: Body Text / Figure Caption Inconsistency

**L488** (end of §4.4 Multi-Seed body text):
> "These low standard deviations relative to the inter-model gap **confirm** that FTMamba's gains are architectural"

**L484** (Figure 7 caption, fixed in R2):
> "The low variance relative to the typical inter-model gap **suggests** that FTMamba's performance is largely architectural; at select horizons ... seed-to-seed variance overlaps"

The body text still uses "confirm" while the caption was already fixed to "suggests." The body text (L488) should be aligned with the more cautious caption language.

**Fix:** Change L488 "confirm" → "suggest" and add the hedge about select horizons.

---

## R3-3 (Minor): ETTh1 Multi-Seed at 720 — Statistical Significance

The Discussion (§5.6 Training Variance) notes that at ETTh1/720, FTMamba's multi-seed mean (0.5033 ± 0.0117) trails PatchTST's single-seed (0.4878). However, the Discussion also notes that PatchTST's variance is unknown. Given that FTMamba's std at T=720 is the largest observed (0.0117), and the gap between FTMamba mean and PatchTST single-run is 0.0155, a PatchTST seed with comparable variance could easily swing the comparison either way. This makes the ETTh1/720 single-seed "win" (0.4668 vs 0.4878, a 4.3% gap) particularly fragile — the single-seed result may be an optimistic draw. Consider adding a sentence noting this specific fragility at T=720, since the paper highlights this horizon as the largest gain (L74, L374, L517).

---

## R3-4 (Minor): Ref10 Formatting

L625: `(Distinguished Paper Award)` in parentheses within the reference entry is non-standard for MDPI reference format. Consider either:
- Moving to a footnote or endnote
- Removing it (the COLM venue is sufficient)
- Reformatting as a note at the end of the reference entry

This is a cosmetic issue unlikely to affect acceptance.

---

## R3-5 (Minor): Limitations "Over-Disclosure"

The Limitations section (§5.7) now enumerates 8 distinct limitation categories across ~320 words. While transparency is commendable, some listed items are standard practices in the field rather than genuine limitations of this specific work:

- **10-epoch training budget** (L543): Standard in Time-Series-Library papers; the paper already states convergence was verified.
- **Instance normalization not ablated** (was m3, now resolved): All methods use equivalent normalization; this is no longer a meaningful limitation.

Consider mentioning these as "scope boundaries" rather than "limitations" to avoid suggesting the authors are at fault for following field conventions. Alternatively, trim the Limitations section to the 5–6 most impactful categories.

---

## R3-6 (Cosmetic): Abstract Word Count

The abstract is now approximately 190 words. MDPI's guideline is "up to ~200 words." This is within range, but verify against the journal's exact word-count requirement at time of submission. Some MDPI journals enforce a hard 200-word limit; others are more flexible.

---

## Summary

| # | Severity | Issue | Action |
|---|----------|-------|--------|
| R3-1 | **Critical** | "10/16" → "9/16" counting error | Fix all 4 occurrences |
| R3-2 | High | Body text "confirm" vs caption "suggests" | Align L488 with caption |
| R3-3 | Low | ETTh1/720 fragility note | Optional: add sentence |
| R3-4 | Low | Ref10 award note format | Optional: reformat |
| R3-5 | Low | Limitations over-disclosure | Optional: trim |
| R3-6 | Info | Abstract word count | Verify before submission |

**Verdict:** Fix R3-1 and R3-2, then the paper is ready for submission to MDPI *Electronics*.

---

*End of Round 3 Review*
