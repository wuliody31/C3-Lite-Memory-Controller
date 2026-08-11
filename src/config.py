from __future__ import annotations

from .errors import ConfigurationError

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ConfigurationError):
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
    required = {
        "controller",
        "query_analysis",
        "routing",
        "retrieval",
        "ranking",
        "conflict",
        "selection",
        "confidence",
        "decision",
        "generation",
        "logging",
    }

    missing = required - set(config)

    if missing:
        raise ConfigError(
            "Missing configuration sections: "
            f"{sorted(missing)}"
        )

    for section_name in required:
        if not isinstance(
            config.get(section_name),
            dict,
        ):
            raise ConfigError(
                "Configuration section "
                f"'{section_name}' must be a mapping."
            )

    def weight_total(
        section_name: str,
        keys: list[str],
    ) -> float:
        section = config[section_name]

        missing_keys = [
            key
            for key in keys
            if key not in section
        ]

        if missing_keys:
            raise ConfigError(
                "Missing configuration keys in "
                f"'{section_name}': {missing_keys}"
            )

        try:
            return sum(
                float(section[key])
                for key in keys
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "Configuration weights in "
                f"'{section_name}' must be numeric."
            ) from exc

    ranking_keys = [
        "lexical_weight",
        "graph_entity_weight",
        "temporal_task_weight",
        "validity_weight",
        "source_confidence_weight",
        "route_compatibility_weight",
    ]

    ranking_total = weight_total(
        "ranking",
        ranking_keys,
    )

    if abs(ranking_total - 1.0) > 1e-6:
        raise ConfigError(
            "Ranking weights must sum to 1.0."
        )

    confidence_keys = [
        "top_evidence_weight",
        "top3_mean_weight",
        "route_confidence_weight",
        "agreement_weight",
    ]

    confidence_total = weight_total(
        "confidence",
        confidence_keys,
    )

    if abs(confidence_total - 1.0) > 1e-6:
        raise ConfigError(
            "Confidence weights must sum to 1.0."
        )

    selection = config["selection"]

    selection_weight_keys = [
        "relevance_weight",
        "coverage_gain_weight",
        "redundancy_weight",
    ]

    if "requirement_weight" in selection:
        selection_weight_keys.insert(
            1,
            "requirement_weight",
        )

    selection_total = weight_total(
        "selection",
        selection_weight_keys,
    )

    if abs(selection_total - 1.0) > 1e-6:
        raise ConfigError(
            "Selection weights must sum to 1.0. "
            f"Found {selection_total:.6f} across "
            f"{selection_weight_keys}."
        )


def frozen_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)
