# ML Extension Roadmap

This note outlines the structured model improvements that follow the baseline MLP forecaster. The goal is to extend the balance-
sheet pipeline with architectures that capture temporal dynamics, quantify uncertainty, and better respect accounting couplings
for both corporates and banks.

## 1. Sequence Modelling

- **Recurrent Encoders:** Train GRU/LSTM encoders over sliding driver windows so the network can internalise non-linear seasonali
  ty without relying on manually engineered lags. Incorporate the bank indicator as an auxiliary channel to preserve sector-
  specific behaviour.
- **Temporal Convolution Networks:** Deploy dilated 1-D convolutions (à la WaveNet/TCN) to capture longer-range dependencies
  while keeping inference parallelisable. Use weight norm and causal convolutions to stabilise training on quarterly data.
- **Transformer Variants:** Evaluate compact encoder-only transformers (e.g., Informer, Autoformer) with relative positional
  encodings for cross-period attention. Constrain output heads to project into driver space before the deterministic projector.

## 2. Probabilistic Forecasting

- **TensorFlow Probability Layers:** Wrap the deterministic projection in a probabilistic head that emits parameterised distri-
  butions (Gaussian or skewed Student-t) for assets and equity. Calibrate via CRPS or negative log-likelihood to obtain estima-
  ted uncertainty bands.
- **Normalising Flows:** Model residuals between projected statements and ground truth using conditional RealNVP/IAF flows to
  better capture heavy-tailed shocks in bank balance sheets.
- **Bayesian Last-Layer Inference:** Place a variational posterior on the final dense layers to obtain uncertainty without re-
  architecting the base network.

## 3. Identity-Aware Architectures

- **Differentiable Reconciliation:** Introduce Lagrange-multiplier layers or differentiable optimisation blocks so the network
  can learn to satisfy auxiliary accounting ratios (e.g., regulatory capital thresholds) beyond the core assets = liabilities +
  equity identity.
- **Hybrid Template Blending:** Learn per-ticker mixture weights between the neural forecast and structural bank templates, with
  monotonic constraints to respect leverage bounds. This generalises the current linear ensemble.

## 4. Training & Evaluation Enhancements

- **Curriculum Learning:** Start training on industrial tickers before introducing banks to stabilise gradients, then fine-tune
  on the full multi-sector set with lower learning rates.
- **Hyperparameter Search:** Use Optuna/Vertex Vizier sweeps over depth, hidden units, and dropout for each architecture, track-
  ing MAE, identity gaps, and net-income alignment.
- **Backtesting:** Expand the evaluation harness to roll forward for multiple periods (walk-forward validation) to capture drift
  in longer horizons.

## 5. Implementation Plan

1. Add GRU-based driver head sharing the deterministic projector; benchmark against the existing MLP using current lagged fea-
   tures.
2. Integrate TensorFlow Probability `DenseVariational` layers to produce forecast intervals; extend the summariser/status CLIs to
   report coverage vs. realised statements.
3. Prototype transformer encoder with learnable accounting constraint penalties.
4. Document findings and decide on the architecture that balances accuracy, interpretability, and computational budget for Part 1
   deliverables.

