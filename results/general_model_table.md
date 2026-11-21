| Prior | eps | Quadratic rel. error (median / max) | Balanced trace rel. error (median / max) |
|---|---|---|---|
| uniform | 0.01 | 0.60% / 2.71% | 34.5% / 254.1% |
| uniform | 0.05 | 3.03% / 15.61% | 31.8% / 280.4% |
| uniform | 0.10 | 5.78% / 28.29% | 31.4% / 304.8% |
| uniform | 0.20 | 9.31% / 51.77% | 29.3% / 376.5% |
| mild | 0.01 | 0.45% / 3.43% | 23.3% / 237.9% |
| mild | 0.05 | 2.25% / 19.92% | 25.9% / 266.7% |
| mild | 0.10 | 4.33% / 36.40% | 27.2% / 294.1% |
| mild | 0.20 | 8.42% / 67.56% | 26.2% / 380.1% |
| skewed | 0.01 | 0.58% / 5.14% | 86.3% / 577.1% |
| skewed | 0.05 | 2.55% / 30.42% | 86.6% / 578.0% |
| skewed | 0.10 | 5.20% / 56.54% | 86.7% / 579.2% |
| skewed | 0.20 | 9.55% / 107.79% | 93.5% / 580.0% |
| dirichlet | 0.01 | 1.02% / 4.38% | 47.8% / 241.6% |
| dirichlet | 0.05 | 5.40% / 25.75% | 45.1% / 275.7% |
| dirichlet | 0.10 | 9.93% / 47.57% | 42.7% / 308.1% |
| dirichlet | 0.20 | 18.81% / 89.94% | 39.5% / 407.7% |

Operating mismatch t = 0.8, sweep over 21 alphas:

| Prior | Channel | Mismatch | eps | Quadratic max rel. err | Balanced max rel. err | alpha* exact / quadratic / balanced |
|---|---|---|---|---|---|---|
| uniform | symmetric_kron | optimistic | 0.366 | 0.0% | 0.0% | 0.00 / 0.00 / 0.00 |
| uniform | symmetric_kron | range_optimistic | 0.205 | 0.0% | 0.0% | 0.00 / 0.00 / 0.00 |
| uniform | asymmetric_kron | optimistic | 0.394 | 8.6% | 2.3% | 0.00 / 0.00 / 0.00 |
| uniform | asymmetric_kron | range_optimistic | 0.230 | 9.8% | 75.9% | 0.00 / 0.00 / 0.00 |
| uniform | random_dirichlet | optimistic | 0.706 | 10.3% | 29.8% | 1.00 / 1.00 / 1.00 |
| mild | symmetric_kron | optimistic | 0.366 | 6.0% | 9.7% | 1.00 / 1.00 / 0.00 |
| mild | symmetric_kron | range_optimistic | 0.205 | 8.1% | 31.7% | 0.00 / 0.00 / 0.00 |
| mild | asymmetric_kron | optimistic | 0.394 | 23.0% | 28.9% | 0.00 / 0.00 / 0.00 |
| mild | asymmetric_kron | range_optimistic | 0.230 | 25.0% | 96.0% | 0.00 / 0.00 / 0.00 |
| mild | random_dirichlet | optimistic | 0.706 | 4.5% | 51.4% | 1.00 / 1.00 / 1.00 |
| skewed | symmetric_kron | optimistic | 0.366 | 16.5% | 50.9% | 0.50 / 0.50 / 0.00 |
| skewed | symmetric_kron | range_optimistic | 0.205 | 17.1% | 113.7% | 0.00 / 0.00 / 0.00 |
| skewed | asymmetric_kron | optimistic | 0.394 | 31.7% | 124.5% | 0.00 / 0.00 / 0.00 |
| skewed | asymmetric_kron | range_optimistic | 0.230 | 34.3% | 204.3% | 0.00 / 0.00 / 0.00 |
| skewed | random_dirichlet | optimistic | 0.706 | 23.5% | 148.9% | 1.00 / 1.00 / 1.00 |
| dirichlet | symmetric_kron | optimistic | 0.366 | 8.8% | 22.8% | 0.00 / 0.00 / 0.00 |
| dirichlet | symmetric_kron | range_optimistic | 0.205 | 1.0% | 23.0% | 0.00 / 0.00 / 0.00 |
| dirichlet | asymmetric_kron | optimistic | 0.394 | 6.9% | 84.8% | 0.00 / 0.00 / 0.00 |
| dirichlet | asymmetric_kron | range_optimistic | 0.230 | 8.9% | 76.5% | 0.00 / 0.00 / 0.00 |
| dirichlet | random_dirichlet | optimistic | 0.706 | 17.4% | 82.4% | 1.00 / 1.00 / 1.00 |
