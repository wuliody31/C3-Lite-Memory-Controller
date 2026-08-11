from src.config import ConfigError
from src.contracts import (
    C3Error,
    ConfigurationError,
    GenerationError,
    ObservabilityError,
    RetrievalError,
)
from src.lifecycle import (
    LifecycleError,
    LifecycleInvariantError,
    LifecycleReplayConflict,
    LifecycleStoreError,
)


def test_configuration_error_preserves_value_error_contract() -> None:
    assert issubclass(ConfigurationError, ValueError)
    assert issubclass(ConfigurationError, C3Error)
    assert issubclass(ConfigError, ConfigurationError)
    assert issubclass(ConfigError, ValueError)


def test_runtime_boundary_errors_preserve_runtime_error_contract() -> None:
    for error_type in (
        RetrievalError,
        GenerationError,
        ObservabilityError,
    ):
        assert issubclass(error_type, RuntimeError)
        assert issubclass(error_type, C3Error)


def test_lifecycle_errors_join_c3_hierarchy_without_breaking_runtime_contract() -> None:
    assert issubclass(LifecycleError, RuntimeError)
    assert issubclass(LifecycleError, C3Error)

    assert issubclass(
        LifecycleInvariantError,
        LifecycleError,
    )
    assert issubclass(
        LifecycleReplayConflict,
        LifecycleError,
    )
    assert issubclass(
        LifecycleStoreError,
        LifecycleError,
    )


def test_resource_cleanup_error_preserves_runtime_contract() -> None:
    from src.contracts import ResourceCleanupError

    assert issubclass(
        ResourceCleanupError,
        RuntimeError,
    )
    assert issubclass(
        ResourceCleanupError,
        C3Error,
    )
