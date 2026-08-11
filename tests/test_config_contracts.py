from copy import deepcopy
from pathlib import Path

import pytest

from src.config import (
    ConfigError,
    load_config,
    validate_config,
)
from src.errors import ConfigurationError


ROOT = Path(__file__).resolve().parents[1]


def valid_config():
    return load_config(
        ROOT / "configs/c3_lite_v2_2_final.yaml"
    )


def test_missing_required_section_is_configuration_error() -> None:
    config = deepcopy(valid_config())
    del config["ranking"]

    with pytest.raises(
        ConfigError,
        match="Missing configuration sections",
    ) as captured:
        validate_config(config)

    assert isinstance(
        captured.value,
        ConfigurationError,
    )


def test_section_must_be_mapping() -> None:
    config = deepcopy(valid_config())
    config["ranking"] = None

    with pytest.raises(
        ConfigError,
        match="must be a mapping",
    ):
        validate_config(config)


def test_missing_nested_weight_is_configuration_error() -> None:
    config = deepcopy(valid_config())
    del config["ranking"]["lexical_weight"]

    with pytest.raises(
        ConfigError,
        match="Missing configuration keys",
    ):
        validate_config(config)


def test_non_numeric_weight_is_configuration_error() -> None:
    config = deepcopy(valid_config())
    config["confidence"][
        "agreement_weight"
    ] = "not-a-number"

    with pytest.raises(
        ConfigError,
        match="must be numeric",
    ):
        validate_config(config)
