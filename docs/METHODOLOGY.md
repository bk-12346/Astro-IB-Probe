# Methodology: The Full Story

This document walks through the analysis as it actually happened, including the dead ends and the reasoning behind each pivot — not just the final results. We think this matters: how we got to the final findings is as informative as the findings themselves, and it's the part a static results table can't show.

## 1. Quantitative spine: linear probing

Before attempting any information-theoretic analysis, we established a real, comparable accuracy metric. For each of AstroPT's layers, in both the 015M and 095M size, we trained a linear probe (logistic regression, class-balanced) against five Galaxy Zoo morphology labels, using AstroPT's own patch tokenization and normalization pipeline (verified directly against the model's source, not assumed).

**Result:** four of five labels show a clear, largely monotonic rise in balanced accuracy with depth. One label — edge-on orientation — never rises meaningfully above chance at any layer, in either model size. This became a genuine finding in its own right, not a bug to chase.

We deliberately built this before any mutual information (MI) work, so that regardless of how the MI analysis turned out, we would have a real, defensible result in hand.

## 2. First MI attempt: discrete clustering (K-means)

Following the classical Tishby/Shwartz-Ziv formulation, we treated the input X as a discrete uniform distribution over our N samples. Since each layer's representation T is a deterministic function of X, I(X;T) reduces to H(T) — computable by discretizing T via K-means clustering and measuring the entropy of the resulting cluster assignment distribution.

**Result:** I(X;T) sat flat, at roughly 87–94% of its theoretical maximum, across every layer, in both model sizes, at every K value tested (10, 30, 50). I(T;label) also showed no clean depth-dependent trend — a contrast with what the linear probe had already found.

This result was ambiguous on its own: it could mean the representations genuinely don't compress, or it could mean K-means-based discretization simply lacks the resolution to detect structure in ~3000 samples spread across 384–768 dimensions (a known curse-of-dimensionality failure mode for this class of estimator).

## 3. Second attempt: PCA-based continuous estimation

To rule out the possibility that discrete clustering specifically was the bottleneck, we tried a structurally different approach: reducing both a raw-input proxy (mean-pooled, normalized patch tensors) and each layer's representation to a handful of principal components, then estimating MI directly via `sklearn`'s kNN-based continuous estimators.

**Result:** I(X;T) no longer saturated — it moved directionally, ranging from ~0.39 to ~0.63 nats and, if anything, showing a mild *increase* with depth in the 095M model. This ruled out "the estimator simply can't move" as an explanation, but I(T;label) under this method remained noisy and non-monotonic, still in visible contrast to the probe's clean trend.

We note an honest limitation of this method: summing per-component MI across PCA dimensions is an approximation to true joint MI (it ignores inter-component redundancy) and likely overestimates it. We report it as such, not as an exact joint estimate.

## 4. Third attempt: linear-Gaussian estimation

The discrepancy between the probe's clean signal and both unsupervised MI methods' noisy results pointed to a specific hypothesis: unsupervised density estimation, at this sample size and dimensionality, may simply lack the power to find a signal concentrated along one specific, task-relevant direction — exactly the direction a *supervised* linear probe is built to find.

To test this, we estimated I(T;label) using the classical linear-Gaussian channel formula (I ≈ -0.5·log(1-ρ²)) applied to the correlation between each layer's probe decision score and the true label.

**Result:** this method recovered the same clean, depth-dependent rise the linear probe had found — for every label showing signal in the probe, and correctly near-zero for edge-on orientation. This resolved the discrepancy: the signal was present in the representations the entire time; the earlier unsupervised methods weren't powerful enough, at this sample size, to find it without supervision.

## 5. The remaining question: is the flat I(X;T) result trustworthy?

At this point we had good evidence that our estimators *could* recover known signal (fitting/I(T;label)) but were still uncertain whether the flat I(X;T) result (no compression) reflected a real absence in AstroPT, or an estimator that simply cannot detect compression under any circumstances — a different, more basic limitation than the label-signal question we'd just resolved.

## 6. Positive control: a tanh MLP

To test this directly, we trained a small tanh-activation MLP to near-zero training loss on the strongest available label (`artefact`), using AstroPT's own 095M final-layer embeddings as a fixed, known-informative input. We then applied the *exact same* K-means-based I(X;T) pipeline used on AstroPT to this network's layers.

**Result:** a clear, sharp compression signature — I(X;T) stayed high and roughly flat through the first several layers, then dropped sharply in the final two (from ~94.9% to 85.2% to 63.3% of theoretical maximum). This directly replicates the tanh-saturation-driven compression described in Saxe et al. (2018), using the identical measurement pipeline that showed AstroPT to be flat.

**Conclusion:** since the same estimator clearly detects compression when it's genuinely present, AstroPT's flat I(X;T) trajectory is evidence of absence, not merely absence of evidence.

## Why we stopped here

Three independent MI estimation approaches, one validated positive control, and one supervised cross-check together give a well-triangulated picture. A fourth estimator would have diminishing returns relative to the time available, and risked estimator-shopping — running more methods until one produces a more dramatic-sounding result, rather than reporting what the evidence actually shows.
