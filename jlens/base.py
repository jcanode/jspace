"""Abstract J-lens backend interface.

A backend turns a prompt into a *slice* analysis: for every (layer, position)
pair it reports the workspace activity and a ranked "readout" of the vocabulary
tokens the model is poised to verbalise there. Both the mock and the (optional)
real transformers backend implement this shape so the frontend is agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass
class Readout:
    """Top-k vocabulary tokens at a single (layer, position) cell."""
    tokens: List[str]
    weights: List[float]

    def as_dict(self) -> Dict:
        return {"tokens": self.tokens, "weights": [round(w, 4) for w in self.weights]}


@dataclass
class Analysis:
    prompt: str
    model: str
    tokens: List[str]
    n_layers: int
    # activity[layer][position] -> 0..1 workspace engagement
    activity: List[List[float]] = field(default_factory=list)
    # readouts[layer][position] -> Readout
    readouts: List[List[Readout]] = field(default_factory=list)
    # region label per layer: "sensory" | "workspace" | "motor"
    layer_regions: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "tokens": self.tokens,
            "n_layers": self.n_layers,
            "activity": [[round(a, 4) for a in row] for row in self.activity],
            "readouts": [[r.as_dict() for r in row] for row in self.readouts],
            "layer_regions": self.layer_regions,
        }


class Backend:
    name = "base"
    display_name = "Base"
    n_layers = 12

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def analyze(self, prompt: str) -> Analysis:
        raise NotImplementedError

    def steer(self, prompt: str, concept: str, strength: float) -> Dict:
        raise NotImplementedError

    def search(self, query: str) -> List[Dict]:
        raise NotImplementedError

    def feature(self, concept: str) -> Dict:
        raise NotImplementedError
