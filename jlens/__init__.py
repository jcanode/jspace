"""J-Space: a mock Jacobian-lens / global-workspace explorer engine."""

from .base import Analysis, Backend, Readout
from .mock import MockBackend

__all__ = ["Analysis", "Backend", "Readout", "MockBackend", "get_backend"]


def get_backend(model: str = "mock", **kwargs) -> Backend:
    """Factory. `model='mock'` (default) always works with zero dependencies.

    Any other value is treated as a HuggingFace model id and routed to the
    optional transformers backend, falling back to the mock if torch is absent.
    """
    if model in (None, "", "mock"):
        return MockBackend(**kwargs)
    try:
        from .hf import TransformersBackend
        return TransformersBackend(model_id=model, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[jspace] real backend unavailable ({exc}); using mock.")
        return MockBackend()
