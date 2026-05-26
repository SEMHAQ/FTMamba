# Reviewer 2 (Round 1)

**Quality of English Language:** The English could be improved to more clearly express the research.

| Criteria | Yes | Can be improved | Must be improved | Not applicable |
|---|---|---|---|---|
| Does the introduction provide sufficient background and include all relevant references? | x |  |  |  |
| Is the research design appropriate? |  | x |  |  |
| Are the methods adequately described? |  | x |  |  |
| Are the results clearly presented? |  | x |  |  |
| Are the conclusions supported by the results? | x |  |  |  |
| Are all figures and tables clear and well-presented? |  | x |  |  |

## Comments and Suggestions for Authors

This paper presents a dual-branch architecture combining a Mamba temporal branch with learnable frequency-domain filtering, fused via an input-dependent gate. The empirical results on four benchmarks are solid, and the ablation study—showing that the gating mechanism contributes more than the frequency branch itself—is a valuable insight. The linear complexity analysis is clear. I recommend **minor revision**.

- When discussing adaptive, input-dependent weighting mechanisms (e.g., in Section 3.5 on gated fusion or Section 5.2 on the importance of gating), the authors should naturally discuss Zhang et al., "Horizontal Multi-Party Data Publishing Under Differential Privacy Via Weight-Aware Bidirectional Generative Adversarial Networks," IEEE Transactions on Knowledge and Data Engineering (2026). This work also learns to balance multiple information streams adaptively and is relevant to the paper's theme of learned fusion weights. Please add this reference in an appropriate methodological context.

No other major issues.

**Submission Date:** 18 May 2026
**Date of Review:** 22 May 2026 08:12:17
