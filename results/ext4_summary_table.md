| Experiment | Ranking agreement (Spearman) | Selected design: Brier vs log | Note |
|---|---|---|---|
| Ext. 1 sweep, range optimistic receiver | 1.0000 | least vulnerable alpha 0.00 vs 0.00 | L_log/L_Brier 1.60-3.27 |
| Ext. 1 sweep, velocity optimistic receiver | 1.0000 | least vulnerable alpha 1.00 vs 1.00 | L_log/L_Brier 1.60-3.27 |
| Ext. 1 J-rule, q=1 | - | max alpha* gap 0.18 over the weight grid | log moves away from the misjudged mode at smaller weights |
| Ext. 1 J-rule, q=0.75 | - | max alpha* gap 0.09 over the weight grid | log moves away from the misjudged mode at smaller weights |
| Ext. 3 probing, A uniform prior | 0.90-1.00 (5 rules, N>=128) | best rule agrees within 2 SE at 7/7 budgets | loss/uniform, two-stage: 0.942 (Brier), 0.947 (log); frequency: 0.990 / 0.991 |
| Ext. 3 probing, B nonuniform prior | 0.40-0.90 (5 rules, N>=128) | best rule agrees within 2 SE at 7/7 budgets | loss/uniform, two-stage: 0.992 (Brier), 0.955 (log); frequency: 1.060 / 1.132 |
| General model, range-optimistic (8 cases) | median 1.000, min +1.000 | least vulnerable alpha agrees 8/8 | max regret: Brier pick under log 0.0%, log pick under Brier 0.0% |
| General model, overconfident (W_hat -> I) (12 cases) | median 0.983, min -0.997 | least vulnerable alpha agrees 6/12 | max regret: Brier pick under log 7.2%, log pick under Brier 44.5% |
| General model, random direction (60 cases) | median 0.984, min +0.047 | least vulnerable alpha agrees 52/60 | max regret: Brier pick under log 11.2%, log pick under Brier 19.2% |
