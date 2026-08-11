# Limitations

Stated plainly.

## Infrastructure constraints

- **850M-parameter model excluded.** Loading the 850M checkpoint crashed the available WSL2 environment on memory. The analysis is limited to 015M and 095M — still a meaningful capacity contrast (>6x parameters), but not the full available scaling range.
- **Training-time trajectory not available.** AstroPT's original release (v1.0.0) includes genuine intermediate training checkpoints, which would have enabled a true training-dynamics analysis rather than a capacity/depth-only one. However, the v1.0.0 PyPI package was found to be non-functional (metadata present, no installable source) during setup, and the checkpoint file format was incompatible with the current (v2.0.x) package's loader. This analysis is therefore a depth × capacity snapshot, not a training-time study.

## Estimation methodology

- **kNN and clustering-based MI estimation is a known-hard problem in high dimensions.** This is not specific to our pipeline — it's an acknowledged, active limitation across the information-bottleneck literature, especially for representations with hundreds of dimensions.
- **PCA-based I(X;T)/I(T;label) is an approximation.** Summing per-component mutual information ignores redundancy between correlated principal components and likely overestimates true joint MI. We report it as an approximate, directional signal, not an exact value.
- **The discrete-sample-identity approach to I(X;T) answers a specific, narrower question** — "how distinguishable/compressed are samples from one another at this layer" — rather than "how much raw visual/pixel content survives." This is the standard classical (Tishby-style) formulation, but it's worth being explicit that it is one valid framing among a few.
- **Only one clustering resolution family (K-means) and one continuous method (PCA + kNN) were tested as unsupervised estimators**, plus one supervised cross-check (linear-Gaussian). A neural estimator (e.g. MINE) might behave differently again, but was ruled out early given CPU-only compute constraints and the project timeline.

## Data and sampling

- **Sample size (3,000 galaxies) is modest for the rarest labels.** Edge-on (~2.8% prevalence) and tight spiral (~1.7% prevalence) have relatively few positive examples in the held-out test split, which likely widens the noise band on their probe and MI estimates specifically, independent of anything about the model.
- **Only one eval subset was used throughout.** Results were not cross-validated across multiple independent samples from the dataset; a different random subset could shift specific numbers, though we would not expect it to reverse the qualitative findings (the fitting trend, the edge-on null result, and the compression absence) given how consistently they held across two model sizes and multiple estimators.

## Scope

- **Two labels (magnitude, stellar mass) from AstroPT's original paper were not tested here** — we focused on the five binary morphology labels for direct comparability with the probe setup and to keep both the probe and MI methodology consistent throughout.
