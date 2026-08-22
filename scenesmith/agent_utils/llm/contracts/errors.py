"""Errors surfaced by the provider-neutral LLM boundary."""


class LLMHarnessError(RuntimeError):
    """Base error surfaced at the provider boundary."""


class LLMCapabilityError(LLMHarnessError):
    """Raised before a run when the selected model cannot satisfy the contract."""


class LLMStructuredOutputError(LLMHarnessError):
    """Raised when deterministic structured-output recovery fails."""


class LLMTimeoutError(LLMHarnessError):
    """Raised when one model turn exceeds the common wall-clock deadline."""


class LLMCircuitOpenError(LLMHarnessError):
    """Raised immediately after a timeout opens the process-local circuit."""
