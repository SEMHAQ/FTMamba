# Reviewer 1 (Round 1)

**Quality of English Language:** The English is fine and does not require any improvement.

| Criteria | Yes | Can be improved | Must be improved | Not applicable |
|---|---|---|---|---|
| Does the introduction provide sufficient background and include all relevant references? | x |  |  |  |
| Is the research design appropriate? | x |  |  |  |
| Are the methods adequately described? |  | x |  |  |
| Are the results clearly presented? |  | x |  |  |
| Are the conclusions supported by the results? |  |  | x |  |
| Are all figures and tables clear and well-presented? |  | x |  |  |

## Comments and Suggestions for Authors

The manuscript proposes FTMamba, a dual-domain architecture for long-term time series forecasting by combining a Mamba-based temporal branch with a learnable FFT-based frequency branch through an adaptive gating mechanism. The topic is timely and relevant, and the general idea is promising. The paper is also mostly clear and readable. However, in my view, the current version still needs stronger methodological and empirical support before the main claims can be fully accepted.

1. **Novelty positioning.** The paper discusses frequency-domain and Mamba-based forecasting models such as FEDformer, FreTS, S-Mamba, and TimeMachine, but these methods are not included in the empirical comparison. Since the proposed model is directly related to both frequency-domain learning and Mamba-based forecasting, comparison with these models would make the contribution much clearer.

2. **Limited experimental setting.** The study uses ETTh1, ETTh2, ETTm1, and Weather datasets. Three of them come from the same ETT family. Additional datasets from different domains, such as Electricity, Traffic, Exchange Rate, Solar, or PEMS, would make the empirical evaluation more convincing.

3. **Cautious interpretation of gains.** FTMamba achieves the best MSE in several cases, but the improvements over strong baselines such as PatchTST and iTransformer are sometimes quite small. Multi-seed results show that some advantages may fall within random seed variation. Soften statements that imply clear superiority and present the model as competitive, with advantages in selected dataset–horizon settings.

4. **Unbalanced multi-seed comparison.** FTMamba is evaluated with multiple seeds, while PatchTST is reported with a single seed. Multi-seed evaluation for the main baselines would strengthen the empirical evidence.

5. **Narrow ablation study.** Conducted only on ETTh1 with prediction horizon 96. Extend ablation to more datasets and horizons. Additional variants (pure Mamba, frequency-only, no frequency and no gate, scalar gate, channel-wise gate, patch-wise gate) would clarify the role of each component.

6. **Tensor-level clarity.** The Mamba block processes variates independently by folding the variate dimension into the batch dimension, but gated fusion mentions broadcast expansion of the temporal output. This creates ambiguity about whether each variable has its own temporal representation. A clearer description of tensor shapes throughout the forward pass would improve methodological clarity.

7. **Additional evaluation metrics.** RMSE, sMAPE, MASE, or NRMSE could provide a more complete picture. Consider statistical comparisons (Diebold–Mariano tests, confidence intervals, or multi-seed significance analysis).

8. **Empirical efficiency support.** Report training time, inference time, throughput, peak GPU memory usage, number of parameters, model size, and FLOPs to support the efficiency claim.

9. **Batch size fairness.** Different batch sizes used in Weather experiments (FTMamba: 8, baselines: 64) may affect training dynamics and efficiency comparison. Discuss more clearly and provide controlled experiments or sensitivity checks.

10. **Gating/frequency interpretation.** Visualizing the learned gate values or frequency filters would help readers understand when the model relies more on temporal information and when it benefits from frequency-domain information.

**Submission Date:** 18 May 2026
**Date of Review:** 21 May 2026 17:09:25
