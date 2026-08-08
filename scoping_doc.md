IB-AstroPT: Final Scoping Doc (Phase 0 close-out)

Title: Information Bottleneck Dynamics Across Scale in an Astronomical Foundation Model

Research question: Does AstroPT's internal representation exhibit an Information Bottleneck–style signature — layers trading input-fidelity for label-relevant compression — as a function of model capacity, and does this vary meaningfully by depth within the network?

Why this is worth doing, not settled ground: The classic IB compression-phase claim is actively disputed — shown to depend on activation function and architecture, with a January 2026 reformulation still finding inconsistent compression behavior under standard IB, and empirical compression patterns recently observed in modern LLM transformers with the causal IB explanation explicitly flagged as unconfirmed. This project brings evidence to that open question in a domain — a scientific, vision-based autoregressive transformer — that hasn't been tested this way.

Direct alignment with the target lab, stated plainly rather than implied: This project isn't just thematically adjacent to FUNDIS — it sits on the specific research line of the coordinating PI, Prof. Slava Voloshynovskiy, whose group's current work centers on information-theoretic bottleneck problems in machine learning, and who has spoken specifically on decomposing mutual information into variational and contrastive components for explainability in self-supervised and generative systems. The write-up should cite 1-2 of his actual papers on bottleneck problems directly in the motivation section — this is meant to read as direct engagement with his published methodology, applied to a domain (astronomy foundation models) his group has only lightly touched via a single current student project.

Final scope, after Phase 0 infrastructure verification:

Model: AstroPT v2.0 (smith42/astropt_v2.0), loaded via the package's own load_astropt() helper — confirmed working end-to-end on CPU
Sizes: 015M and 095M parameters. 850M excluded — confirmed to crash the available WSL2 memory allocation during Phase 0 testing.
No training-step axis. The original paper's release does include genuine intermediate training checkpoints (v1.0.0, smith42/astroPT), which would have enabled a true training-time trajectory — but that checkpoint format requires the v1.0.0 package release, which was found during Phase 0 to be a non-functional PyPI entry (metadata present, no installable source). This is stated as an infrastructure limitation, not a research design flaw.
Analysis axes: model capacity (2 points) × layer depth (per-model layer count) — a real, if modest, grid for the info-plane analysis

Approach, unchanged in method from earlier scoping:

Linear probe accuracy against Galaxy Zoo morphology labels (smooth/disc/artefact/edge-on/spiral) per layer per size — quantitative spine, cross-checked against the original paper's own reported probe results for the same labels
kNN-based mutual information estimation — I(activation; input), I(activation; label) — per layer per size
Info-plane trajectory plotted across depth and compared across the two capacity points; honest reporting regardless of outcome

Known limitations, stated upfront:

Only two capacity points, no training-time dimension — a depth-wise, cross-capacity snapshot rather than a training dynamics study
kNN MI estimation is sensitive to sample size/hyperparameters; single estimator used given timeline
850M excluded on hardware grounds, documented plainly, consistent with how the astro-ib-probe project has handled every other infrastructure constraint so far

Success criteria: Complete, reproducible probe-accuracy grid (guaranteed floor). Info-plane analysis on top is the differentiated contribution regardless of which direction the result points.