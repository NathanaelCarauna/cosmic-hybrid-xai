# Cross-Dataset Heterogeneity Matrix

MAE values. Lower is better. *(home)* = method trained/validated on this dataset.

| Method | De Vito (119) | Zenodo (2040) | Ochodek (437) | User Stories (100) |
| --- | --- | --- | --- | --- |
| E30f-clean (Hybrid-XAI) | 1.133 *(holdout-30 / LOOCV-119=0.933, n=30)* | 1.185 *(cross-dataset (frozen hparams, n=200), n=200)* | 7.400 *(cross-dataset (frozen hparams, n=200), n=200)* | 1.490 *(cross-dataset (frozen hparams, n=98), n=98)* |
| E33d (NER+Ontologia) | 0.933 *(home, n=119)* | 0.345 *(sentence rule R_DISPLAY_IMPLICIT_READ (held-out 70/30), n=2040)* | 4.410 *(per-format UC-name estimator (held-out LOPO-CV), n=437)* | 2.602 *(cross-dataset (US wrapped as pseudo-UC; under-counts implicit E/R), n=98)* |
| E33d+E34a (com adaptador Connextra) | 0.933 *(home, n=119)* | 0.345 *(sentence rule R_DISPLAY_IMPLICIT_READ (held-out 70/30), n=2040)* | 4.410 *(per-format UC-name estimator (held-out LOPO-CV), n=437)* | 0.500 *(test-78 (adaptador Connextra calibrado em dev-20; full-98: MAE=0.469, SA=0.349), n=78)* |
| R1-COSMET (gpt-4-turbo) | 0.370 *(home (gpt-4-turbo, per-project weighted avg), n=119)* | 0.775 *(cross-dataset (n=200, gpt-4-turbo), n=200)* | 6.700 *(cross-dataset (n=200, gpt-4-turbo), n=200)* | 0.888 *(cross-dataset (n=98, gpt-4-turbo), n=98)* |
| R2-DEEP-COSMIC (RF) | 3.092 *(zero-shot (train=Ochodek, RF name-only))* | 5.809 *(zero-shot (train=Ochodek, RF sentence))* | 3.591 *(home (RF canonical), n=437)* | N/A (format) |
| R3-HKO (Hussain) | 1.698 *(LOOCV (UC-flow parsed as sentences), n=119)* | 0.289 *(n=2040)* | 4.270 *(home, n=437)* | 0.602 *(home (Hussain designed for user stories), n=98)* |

**Legend:** TBD = not yet evaluated; N/A (format) = dataset format incompatible with method.

## Standardized Accuracy (SA)

| Method | De Vito (119) | Zenodo (2040) | Ochodek (437) | User Stories (100) |
| --- | --- | --- | --- | --- |
| E30f-clean (Hybrid-XAI) | 0.285 | -1.756 | -0.475 | -1.065 |
| E33d (NER+Ontologia) | 0.555 | 0.161 | 0.067 | -2.606 |
| E33d+E34a (com adaptador Connextra) | 0.555 | 0.161 | 0.067 | 0.261 |
| R1-COSMET (gpt-4-turbo) | 0.824 | -0.802 | -0.336 | -0.230 |
| R2-DEEP-COSMIC (RF) | -0.475 | -13.126 | 0.240 | N/A (format) |
| R3-HKO (Hussain) | 0.190 | 0.297 | 0.096 | 0.166 |
