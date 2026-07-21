from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"controller", "query_analysis", "routing", "retrieval", "ranking", "conflict", "selection", "confidence", "decision", "generation", "logging"}
    missing = required - set(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")
    ranking_keys = ["lexical_weight", "graph_entity_weight", "temporal_task_weight", "validity_weight", "source_confidence_weight", "route_compatibility_weight"]
    if abs(sum(float(config["ranking"][k]) for k in ranking_keys) - 1.0) > 1e-6:
        raise ConfigError("Ranking weights must sum to 1.0.")
    confidence_keys = ["top_evidence_weight", "top3_mean_weight", "route_confidence_weight", "agreement_weight"]
    if abs(sum(float(config["confidence"][k]) for k in confidence_keys) - 1.0) > 1e-6:
        raise ConfigError("Confidence weights must sum to 1.0.")
    selection = config["selection"]
    total = float(selection["relevance_weight"]) + float(selection["redundancy_weight"]) + float(selection["coverage_gain_weight"])
    if abs(total - 1.0) > 1e-6:
        raise ConfigError("Selection weights must sum to 1.0.")


def frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)
