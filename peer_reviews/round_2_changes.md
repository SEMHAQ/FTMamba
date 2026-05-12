# Round 2 Revision Change Log

**Date:** 2026-05-12
**Target Journal:** MDPI *Electronics* (IF 2.6)
**Review:** peer_reviews/round_2_review.md

---

## Changes Implemented

### R2-1: Section Rename (High → done)
- **"Materials and Methods" → "Methods"** (L100)
- Rationale: purely computational work, no physical materials. Aligns with MDPI conventions for CS papers.

### R2-2: Reference Verification & Update (High → done)
Three references were verified and updated from arXiv-only to their peer-reviewed venues:

| Ref | Old | New |
|-----|-----|-----|
| ref10 (Mamba) | arXiv:2312.00752 | **COLM 2024**, Distinguished Paper Award |
| ref11 (S-Mamba) | arXiv:2403.11144 (Wang, Zhang, Li) | **Neurocomputing 2025**, 619, 129178 (Wang, Kong, Feng et al., 8 authors) |
| ref12 (TimeMachine) | arXiv:2403.03820 (Chen, Li, Wang) | **ECAI 2024**, FAIA Vol. 392, pp. 1688-1695, DOI: 10.3233/FAIA240677 (Ahamed & Cheng) |

Note: ref11 author list was corrected (original had 3 incorrect authors; actual paper has 8).

### R2-6: Multi-Seed Figure Caption (High → done)
- **Fig. 7 caption** now reads: "...suggests that FTMamba's performance is largely architectural; at select horizons (ETTh1/96, ETTh1/720), seed-to-seed variance overlaps with inter-model differences (see Section 5)."
- Previously stated "confirms" — now aligned with the qualified discussion text.

### R2-7: Spectral Dilution Qualifier (Medium → done)
- Changed from factual statement to hypothesis: "We hypothesize that ETTm1's 15-minute resolution spreads energy..."
- Added: "This hypothesis could be tested through spectral analysis of the datasets ... which we leave to future work."
- Explicitly named "spectral dilution" as a coined term requiring verification.

### R2-8: Practical Deployment Considerations (Medium → done)
- Added ~200-word subsection in §Computational Efficiency covering:
  - Reuse of optimized CUDA kernels (Mamba selective scan) and cuFFT libraries
  - Modest parameter overhead of frequency branch + gate
  - Channel-independent design = linear memory scaling with variate count
  - Edge/real-time deployment relevance
  - Acknowledgment that empirical profiling remains future work
- Targeted at *Electronics* readership (practitioners, applied researchers)

### R2-5: Publisher's Note (Low → done)
- Previously `\PublishersNote{}` (empty)
- Now: "This article has been submitted to *Electronics* and is currently under peer review. The published version will be available on the journal's website upon acceptance."

---

## Items NOT Changed (Low Priority / Optional)

### R2-3: Figure Resolution
The paper uses vector PDF figures (ideal for MDPI). Associated PNG files exist in the figures directory. Before final submission, regenerate PNGs at 600 dpi using matplotlib (`plt.savefig(dpi=600)`). Vector PDFs are preferred by MDPI and should be used in the final upload.

**Action:** `plot_results.py` and `plot_additional.py` should use `dpi=600` when exporting PNG fallbacks.

### R2-4: Graphical Abstract
MDPI encourages (does not require) a graphical abstract. The architecture diagram (architecture.pdf) could be adapted with annotation of the key finding. Optional — can be added during final submission.

---

## Summary

7 text-level changes applied to FTMamba.tex. Estimated 1.5 hours of work. The paper is now calibrated for MDPI *Electronics* submission with:
- Correctly verified peer-reviewed references (no arXiv-only citations for key methods)
- Qualified, honest claims throughout
- Practical deployment discussion for the journal's applied readership
- MDPI-compliant formatting (highlights, publisher's note, section naming)
