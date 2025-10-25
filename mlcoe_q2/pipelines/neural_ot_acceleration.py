"""Train a neural network to accelerate OT-based resampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mlcoe_q2.models.resampling.neural_ot import (
    NeuralOTConfig,
    train_neural_ot_accelerator,
)
from mlcoe_q2.utils import add_config_argument, ensure_output_paths, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--epsilon-range", type=float, nargs=2, default=(0.05, 0.5))
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-units", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--sinkhorn-iters", type=int, default=30)
    parser.add_argument("--sinkhorn-tolerance", type=float, default=1e-4)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/artifacts/neural_ot_acceleration.json"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("reports/artifacts/models/neural_ot.keras"),
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/bonus_neural_ot.md"),
    )
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def _write_json(path: Path, config: NeuralOTConfig, metrics: dict[str, float]) -> None:
    payload = {
        "config": asdict(config),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_markdown(path: Path, config: NeuralOTConfig, metrics: dict[str, float]) -> None:
    lines = [
        "# Bonus — Neural OT Resampling",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Train loss | {metrics['train_loss']:.4f} |",
        f"| Validation loss | {metrics['val_loss']:.4f} |",
        f"| Test loss | {metrics['test_loss']:.4f} |",
        f"| Test MAE | {metrics['test_mae']:.4f} |",
        f"| Row normalisation error | {metrics['row_normalization_error']:.4e} |",
        f"| Transport L1 error | {metrics['plan_l1_error']:.4e} |",
        f"| Per-sample Sinkhorn time (s) | {metrics['per_sample_sinkhorn']:.5f} |",
        f"| Per-sample neural time (s) | {metrics['per_sample_neural']:.5f} |",
        f"| Relative speedup | {metrics['relative_speedup']:.2f}× |",
        "",
        "## Configuration",
        "",
        f"- Particles: {config.num_particles}",
        f"- State dimension: {config.state_dim}",
        f"- Samples: {config.num_samples}",
        f"- Hidden units: {list(config.hidden_units)}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m mlcoe_q2.pipelines.neural_ot_acceleration --config configs/q2/neural_ot_acceleration.json",
        "```",
    ]
    path.write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = NeuralOTConfig(
        num_particles=args.num_particles,
        state_dim=args.state_dim,
        epsilon_range=tuple(args.epsilon_range),
        num_samples=args.num_samples,
        validation_split=args.validation_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=tuple(args.hidden_units),
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tolerance=args.sinkhorn_tolerance,
        random_seed=args.random_seed,
    )

    ensure_output_paths([args.output_json, args.status_md, args.model_path])
    metrics = train_neural_ot_accelerator(
        config,
        model_dir=args.model_path,
    )
    _write_json(args.output_json, config, metrics)
    _write_markdown(args.status_md, config, metrics)


if __name__ == "__main__":
    main()
