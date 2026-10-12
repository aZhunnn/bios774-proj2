# UCI Wine Quality White-Wine Source and Preprocessing

Source: UCI Machine Learning Repository, Wine Quality dataset, official `winequality-white.csv`.

License: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated by UCI.

Citation: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

Preprocessing decisions:
- The 11 physicochemical variables are modeled as continuous tabular features.
- `quality` is recorded as a target role and excluded from unconditional generation.
- The raw semicolon-delimited CSV is loaded directly; no rows or feature columns are dropped.
- Missing values are rejected by the workflow; this dataset has no missing values in the downloaded file.
- Splitting is deterministic and preprocessing statistics are fit on the training split only.
- Validation data is used for model selection; held-out test data is used only for final evaluation.
