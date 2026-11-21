Setting: Extension 1 baseline (uniform prior, W = B(0.1) kron B(0.1)), range-optimistic receiver (p_hat_r = 0.02, q = 1). Every row is evaluated in this same setting.
Required probe budget: known-token probes (sensitivity-aware allocation) so that a receiver using the estimated channel has expected excess Brier loss <= 0.0001. Predicted by the quadratic approximation; the last column is the Monte Carlo mean at that budget (1000 runs) divided by the target.

| Criterion | Ext. | alpha | Hard accuracy | R_task | L_mis | L_mis vs info-only | L_log | Required probes | MC loss / target |
|---|---|---|---|---|---|---|---|---|---|
| max hard accuracy | Ext. 1 | 0.50 | 0.5250 | 0.1110 | 0.00242 | +0% | 0.00545 | 2,394 | 1.00 +- 0.02 |
| max R_task (information only) | Ext. 1 | 0.50 | 0.5250 | 0.1110 | 0.00242 | +0% | 0.00545 | 2,394 | 1.01 +- 0.02 |
| max V_prob = R_task - L_mis | Ext. 1 | 0.48 | 0.5249 | 0.1110 | 0.00237 | -2% | 0.00525 | 2,393 | 0.99 +- 0.01 |
| max J_lambda, lambda = 5 | Ext. 1 | 0.39 | 0.5229 | 0.1102 | 0.00210 | -13% | 0.00437 | 2,368 | 0.98 +- 0.02 |
| max J_lambda, lambda = 10 | Ext. 1 | 0.24 | 0.5128 | 0.1061 | 0.00157 | -35% | 0.00295 | 2,250 | 0.99 +- 0.02 |
| max J_lambda, lambda = 20 | Ext. 1 | 0.00 | 0.4688 | 0.0927 | 0.00052 | -79% | 0.00083 | 1,860 | 1.00 +- 0.02 |
| min L_mis (reliability only) | Ext. 1 | 0.00 | 0.4688 | 0.0927 | 0.00052 | -79% | 0.00083 | 1,860 | 0.99 +- 0.02 |
| min spectral worst case, eps = 0.1 | Ext. 2 | 0.50 | 0.5250 | 0.1110 | 0.00242 | +0% | 0.00545 | 2,394 | 1.00 +- 0.02 |
| min realizable worst case, doubly-stochastic, eps = 0.1 | Ext. 2 | 0.50 | 0.5250 | 0.1110 | 0.00242 | +0% | 0.00545 | 2,394 | 1.01 +- 0.02 |
| min realizable worst case, row-stochastic, eps = 0.1 | Ext. 2 | 0.35 | 0.5211 | 0.1094 | 0.00197 | -19% | 0.00398 | 2,346 | 0.99 +- 0.02 |
| max I(X;M) - lambda L_log, weight matched to lambda = 10 | Ext. 4 | 0.05 | 0.4827 | 0.0962 | 0.00076 | -69% | 0.00126 | 1,962 | 0.99 +- 0.02 |
| min L_log | Ext. 4 | 0.00 | 0.4688 | 0.0927 | 0.00052 | -79% | 0.00083 | 1,860 | 0.99 +- 0.02 |

Sensitivity-aware probing needs at most 0.0% fewer probes than uniform here because the baseline channel is token-symmetric; the saving appears for token-dependent channels (Ext. 3, Fig. 7b).
