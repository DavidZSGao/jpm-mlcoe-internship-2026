"""Helper utilities for loading CLI configuration overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


_JSON_SUFFIXES = {".json"}
_YAML_SUFFIXES = {".yaml", ".yml"}


class ConfigError(ValueError):
    """Raised when a CLI configuration file is invalid."""


def _load_raw_config(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in _JSON_SUFFIXES:
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in _YAML_SUFFIXES:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigError(
                "YAML configuration requested but PyYAML is not available"
            ) from exc

        return yaml.safe_load(path.read_text(encoding="utf-8"))

    # Default to JSON parsing to avoid surprising behaviour.
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> Mapping[str, Any]:
    """Load a configuration mapping from a JSON or YAML file."""

    payload = _load_raw_config(path)
    if not isinstance(payload, Mapping):
        raise ConfigError("Configuration file must contain a mapping/dictionary")
    return payload


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "t", "yes", "y", "1"}:
            return True
        if normalised in {"false", "f", "no", "n", "0"}:
            return False
    raise ConfigError(f"Cannot coerce value {value!r} to boolean")


def _coerce_sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_value(value: Any, template: Any) -> Any:
    template_type = template if isinstance(template, type) else type(template)

    if template is None and not isinstance(template, type):
        template_type = type(None)

    if template_type is bool:
        return _coerce_bool(value)

    if isinstance(template_type, type) and issubclass(template_type, Path):
        if value is None:
            return None
        return Path(value).expanduser()

    if template_type is int:
        return int(value)

    if template_type is float:
        return float(value)

    if template_type is list:
        return _coerce_sequence(value)

    if template_type is tuple:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)

    if template_type is type(None):
        return value

    if template_type is str:
        return str(value)

    return value if isinstance(value, template_type) else template_type(value)


def apply_cli_config(
    namespace: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    type_overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Apply configuration overrides onto an argparse namespace."""

    overrides = dict(type_overrides or {})
    for key, value in config.items():
        if key == "config":
            continue
        if not hasattr(namespace, key):
            raise ConfigError(f"Unknown CLI option in configuration: {key}")
        template = overrides.get(key, getattr(namespace, key))
        coerced = _coerce_value(value, template)
        setattr(namespace, key, coerced)
    return namespace


def load_cli_overrides(
    namespace: argparse.Namespace,
    config_path: Path,
    *,
    type_overrides: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Load and normalise configuration overrides for argparse defaults."""

    config = load_config(config_path)
    working = argparse.Namespace(**vars(namespace))
    apply_cli_config(working, config, type_overrides=type_overrides)
    return {
        key: getattr(working, key)
        for key in config
        if key != "config" and hasattr(namespace, key)
    }


def add_config_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = "Optional JSON/YAML file providing default CLI arguments",
) -> None:
    """Register a ``--config`` argument on the provided parser."""

    parser.add_argument("--config", type=Path, help=help_text)


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    type_overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments while honouring optional configuration files.

    The parser **must** include a ``--config`` option registered via
    :func:`add_config_argument`. If the option is supplied, its values are loaded
    and applied as defaults prior to the final parse, allowing explicit command
    line arguments to take precedence.
    """

    toggled: list[tuple[argparse.Action, bool]] = []
    for action in parser._actions:
        if getattr(action, "required", False):
            toggled.append((action, True))
            action.required = False

    try:
        preliminary, _ = parser.parse_known_args(argv)
    finally:
        for action, was_required in toggled:
            action.required = was_required

    config_path = getattr(preliminary, "config", None)
    override_required: list[tuple[argparse.Action, bool]] = []
    if config_path is not None:
        overrides = load_cli_overrides(
            preliminary,
            config_path,
            type_overrides=type_overrides,
        )
        parser.set_defaults(**overrides)
        for action in parser._actions:
            if getattr(action, "dest", None) in overrides and getattr(action, "required", False):
                override_required.append((action, True))
                action.required = False

    try:
        return parser.parse_args(argv)
    finally:
        for action, was_required in override_required:
            action.required = was_required
