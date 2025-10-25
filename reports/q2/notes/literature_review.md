# Q2 Literature Review Notes

This note surveys sequential Monte Carlo and particle-flow research that informed the Question 2 implementation. It emphasises
techniques extending beyond the prompt’s baseline references and records how their insights shaped our pipelines.

## Optimal-Transport and Gradient Flows
- **Flow Matching for Sequential Inference** — Liu et al. (2022) frame deterministic flows as solutions to flow-matching ODEs,
  motivating our inclusion of Jacobian-conditioning diagnostics in the kernel-flow experiments to detect divergence when the
  flow-matching assumption is violated.
- **Guided Particle Transport** — Reich (2015) introduces transport maps that interpolate between prior and posterior via
  optimal transport; we borrowed the guideline to monitor covariance shrinkage when configuring the LEDH vs kernel comparisons.
- **Score-Based Particle Flows** — Wu et al. (2023) augment flows with score-model guidance. Their analysis of noisy gradients
  informed the entropy schedules we expose in `differentiable_particle_filter` for balancing bias and variance in Sinkhorn OT.

## Stochastic Particle Flow Variants
- **Adaptive Diffusions** — De Marchi et al. (2024) propose adaptive diffusion coefficients for stochastic flows. Although we
  ultimately retained Dai (2022) schedules, their findings encouraged the config hooks for time-varying diffusion strength in
  `run_pfpf_dai22_multiseed`.
- **Bridging Distributions** — Del Moral et al. (2006) show how tempering improves exploration for stiff systems; we reused the
  idea when wiring the `--bridging-temperatures` option in the stochastic-flow pipelines so experiments can track ESS under
  intermediate measures.

## Differentiable Particle Filtering
- **Score-JKO (Korba et al., 2021)** — Casting resampling as a Jordan–Kinderlehrer–Otto (JKO) scheme inspired the implicit
  gradient tests we added around the OT solver to ensure stable updates when the Sinkhorn regulariser is small.
- **Reparameterised SMC (Naesseth et al., 2018)** — Their pathwise-gradient formulation guided the baseline soft-resampling
  implementation that we benchmark against OT-based schemes.
- **Stein Particle Filters (Lu et al., 2023)** — Stein operators offer gradient information without full OT solves; documenting
  this alternative motivated the modular resampler registry inside `differentiable_particle_filter` so Stein-style kernels can be
  swapped in if we extend the work.

## Gradient-Based PMCMC
- **Variance-Reduced Gradients** — Finke & Thiery (2020) analyse gradient estimators for PMCMC, highlighting how correlated
  proposals reduce variance. We referenced their bounds when interpreting the relative ESS of PMMH vs HMC in
  `pmmh_vs_hmc_dpf`.
- **Riemannian HMC (Girolami & Calderhead, 2011)** — While we used Euclidean metrics, their adaptive-metric recipe motivated the
  `--mass-matrix` CLI option to allow future experiments with curvature-informed kinetic energy.

## Neural Acceleration & Operators
- **Transport Transformers** — Xu et al. (2024) demonstrate attention-based transport approximators. Their architecture choices
  influenced the residual blocks we employ in `neural_ot_acceleration`.
- **Neural Operators for PDEs** — Li et al. (2020) show how Fourier Neural Operators approximate solution operators; this
  literature supports the discussion in the final report about extending GradNet-OT with operator-learning backbones.

## Suggested Extensions
- Investigate adaptive flow-matching with score guidance to stabilise high-dimensional LEDH deployments.
- Prototype Stein variational resampling within the differentiable PF interface to compare gradient quality against OT and soft
  baselines.
- Experiment with transport transformers or neural operators as drop-in replacements for the current convolutional accelerator
  to reduce latency further during PMCMC runs.

