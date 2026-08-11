# Future Work

Concrete extensions that were explicitly scoped out given the project timeline.

## Training-time trajectory

The single most valuable extension: resolving the v1.0.0 package/checkpoint-format incompatibility (documented in `LIMITATIONS.md`) would unlock genuine intermediate training checkpoints, enabling a true training-dynamics IB analysis rather than the current depth × capacity snapshot. This would most directly test the classical fitting-then-compression claim in its original form.

## Larger-scale capacity point

Testing the 850M-parameter model (excluded here on memory grounds) on different hardware — a cloud GPU instance, for instance — would extend the capacity axis and clarify whether the mild depth-dependent I(X;T) trend seen under PCA in 095M continues, plateaus, or reverses at larger scale.

## Extending the positive control

The tanh-MLP positive control was run only against the K-means pipeline. Running the same control through the PCA-based and linear-Gaussian pipelines would confirm (or complicate) the same conclusion — that AstroPT's lack of compression is real — under every estimator used in this project, not just one.

## Neural MI estimation (MINE)

A trained-critic approach to MI estimation could in principle detect structure that both the discrete and PCA-based estimators missed, without needing a proxy or discretization step. This was ruled out early given CPU-only compute and the fixed application deadline, but would be a natural next method to try with more compute available.

## Larger, stratified sample for rare labels

A larger eval subset, or one deliberately oversampled for edge-on and tight-spiral positives, would tighten the noise band on those two labels specifically and clarify whether edge-on's null result holds at higher statistical power, or whether it's partly a sample-size artifact.
