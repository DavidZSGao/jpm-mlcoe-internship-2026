import json
from pathlib import Path

from mlcoe_q1.pipelines import build_driver_dataset, evaluate_forecaster, train_forecaster
from mlcoe_q1.pipelines.evaluate_forecaster import _parse_quantiles


def test_build_driver_dataset_config_parsing(tmp_path):
    data_root = tmp_path / "processed"
    output = tmp_path / "driver.parquet"
    config = {
        "data_root": str(data_root),
        "output": str(output),
        "lags": 2,
        "lag_features": ["sales", "sales_growth"],
        "tickers": ["gm", "msft"],
        "keep_missing_lags": True,
        "log_level": "DEBUG",
    }
    config_path = tmp_path / "build_config.json"
    config_path.write_text(json.dumps(config))

    args = build_driver_dataset.parse_args(["--config", str(config_path)])

    assert args.data_root == data_root
    assert args.output == output
    assert args.lags == 2
    assert args.lag_features == ["sales", "sales_growth"]
    assert args.tickers == ["gm", "msft"]
    assert args.keep_missing_lags is True
    assert args.log_level == "DEBUG"



def test_train_forecaster_config_parsing(tmp_path):
    drivers = tmp_path / "drivers.parquet"
    output_dir = tmp_path / "model"
    processed_root = tmp_path / "processed"
    config = {
        "drivers": str(drivers),
        "output_dir": str(output_dir),
        "processed_root": str(processed_root),
        "epochs": 25,
        "batch_size": 8,
        "learning_rate": 5e-4,
        "architecture": "gru",
        "sequence_length": 5,
        "gru_units": "128,32",
        "recurrent_dropout": 0.2,
        "distribution": "variational",
        "kl_weight": 5e-4,
        "calibrate_banks": True,
        "log_level": "DEBUG",
    }
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps(config))

    args = train_forecaster.parse_args(["--config", str(config_path)])

    assert args.drivers == drivers
    assert args.output_dir == output_dir
    assert args.processed_root == processed_root
    assert args.epochs == 25
    assert args.batch_size == 8
    assert args.learning_rate == 5e-4
    assert args.architecture == "gru"
    assert args.sequence_length == 5
    assert args.gru_units == "128,32"
    assert args.recurrent_dropout == 0.2
    assert args.distribution == "variational"
    assert args.kl_weight == 5e-4
    assert args.calibrate_banks is True
    assert args.log_level == "DEBUG"



def test_evaluate_forecaster_config_parsing(tmp_path):
    drivers = tmp_path / "drivers.parquet"
    model_dir = tmp_path / "model"
    processed_root = tmp_path / "processed"
    output = tmp_path / "eval.parquet"
    config = {
        "drivers": str(drivers),
        "model_dir": str(model_dir),
        "processed_root": str(processed_root),
        "output": str(output),
        "bank_mode": "ensemble",
        "mc_samples": 64,
        "quantiles": [0.2, 0.8],
        "horizon": 3,
        "log_level": "DEBUG",
    }
    config_path = tmp_path / "eval_config.json"
    config_path.write_text(json.dumps(config))

    args = evaluate_forecaster.parse_args(["--config", str(config_path)])

    assert args.drivers == drivers
    assert args.model_dir == model_dir
    assert args.processed_root == processed_root
    assert args.output == output
    assert args.bank_mode == "ensemble"
    assert args.mc_samples == 64
    assert args.horizon == 3
    assert args.log_level == "DEBUG"
    assert _parse_quantiles(args.quantiles) == [0.2, 0.8]

