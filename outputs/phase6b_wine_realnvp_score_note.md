# Phase 6B RealNVP Score Difference Note

The Phase 4A and Phase 6A/6B Wine Quality runs use the same validation split size,
same shared synthetic-vs-validation `selection_score` definition, and the same
validation generation seed for flow (`2039`).

Observed validation scores:
- Phase 4A RealNVP: `0.912417743245997`
- Phase 6A/6B RealNVP: `0.8829950397244746`

The difference is attributable to training configuration and budget context, not
to a changed cross-family evaluation metric:
- Phase 4A compared VAE, RealNVP, and bootstrap under the earlier budget.
- Phase 6A/6B allocates the same global budget across four trained families:
  VAE, RealNVP, GMM, and Gaussian copula.
- The Phase 6A/6B Wine run used one RealNVP candidate with `epochs=5`,
  `hidden_dim=32`, `num_layers=4`, and seed `2127`.
- The shared validation score is computed after generation and does not compare
  raw flow NLL against VAE ELBO or classical likelihoods.

No held-out test metric was used to choose or revise the Phase 6B planning rule.
