"""Stable exception categories for C3 service boundaries."""


class C3Error(Exception):
    """Base class for C3-specific failures."""


class ConfigurationError(ValueError, C3Error):
    """Invalid C3 configuration or construction parameters."""


class RetrievalError(RuntimeError, C3Error):
    """Failure while accessing a memory retrieval backend."""


class GenerationError(RuntimeError, C3Error):
    """Failure while executing a language-model backend."""


class ObservabilityError(RuntimeError, C3Error):
    """Failure inside an observability or telemetry adapter."""
