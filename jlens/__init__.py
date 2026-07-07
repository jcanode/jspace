"""J-Space: a mock Jacobian-lens / global-workspace explorer engine.

Everything here runs **fully offline on the Python standard library** — no
downloads, no `torch`, no `transformers`, no HuggingFace access. Two backends:

* ``mock``   — deterministic synthetic readouts from a hand-curated concept
               graph. Always meaningful, zero setup. (default)
* ``corpus`` — a tiny n-gram language model *learned at startup* from a bundled
               text corpus, so readouts are computed from real data. Still pure
               stdlib and offline.
"""

from .base import Analysis, Backend, Readout
from .mock import MockBackend

__all__ = ["Analysis", "Backend", "Readout", "MockBackend", "get_backend"]


def get_backend(model: str = "mock", **kwargs) -> Backend:
    """Factory. All backends are pure-stdlib and offline.

    * ``model='mock'``   (default) synthetic concept-graph backend.
    * ``model='corpus'`` n-gram model trained on the bundled corpus.
    """
    if model in ("corpus", "ngram"):
        from .corpus import CorpusBackend
        return CorpusBackend(**kwargs)
    if model not in (None, "", "mock"):
        print(f"[jspace] unknown backend '{model}'; using offline mock backend.")
    return MockBackend(**kwargs)
