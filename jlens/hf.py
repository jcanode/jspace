"""Optional real-model J-lens backend for open-source models.

This is a *lightweight approximation* of the Jacobian lens using the logit-lens
trick: project the residual stream at each layer through the model's final
layer-norm + unembedding to read out which vocabulary tokens each intermediate
activation is "poised to verbalise". A true J-lens averages the linearized
(Jacobian) effect of an activation on output logits across contexts; the logit
lens is the single-context, first-order stand-in and is plenty for exploration.

Requires `torch` and `transformers`. If they're missing, importing this module
still succeeds but constructing the backend raises a clear error, so the server
can silently fall back to the mock backend.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .base import Analysis, Backend, Readout
from . import knowledge as kb

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAVE_TORCH = True
except Exception:  # noqa: BLE001
    _HAVE_TORCH = False


class TransformersBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str = "gpt2", top_k: int = 6, device: str = "cpu"):
        if not _HAVE_TORCH:
            raise RuntimeError(
                "transformers backend requires `torch` and `transformers`. "
                "Install them or run the server with the mock backend."
            )
        self.model_id = model_id
        self.display_name = f"{model_id} (logit-lens J-lens)"
        self.top_k = top_k
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, output_hidden_states=True
        ).to(device).eval()
        cfg = self.model.config
        self.n_layers = getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 12))
        # final norm + unembedding for the logit lens
        self._ln_f = self._find_final_norm()
        self._unembed = self.model.get_output_embeddings()

    def _find_final_norm(self):
        m = self.model
        for path in ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return torch.nn.Identity()

    def tokenize(self, text: str) -> List[str]:
        ids = self.tok(text, return_tensors="pt").input_ids[0]
        return [self.tok.decode([i]).replace("\n", "\\n") for i in ids]

    @torch.no_grad()
    def analyze(self, prompt: str) -> Analysis:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        ids = enc.input_ids[0]
        str_tokens = [self.tok.decode([i]).replace("\n", "\\n") for i in ids]
        out = self.model(**enc)
        # hidden_states: tuple(len = n_layers+1) of [1, seq, d]
        hs = out.hidden_states[1:]  # skip embeddings layer
        n_layers = len(hs)
        activity, readouts, regions = [], [], []
        for layer, h in enumerate(hs):
            regions.append(self._region(layer, n_layers))
            logits = self._unembed(self._ln_f(h[0]))  # [seq, vocab]
            probs = torch.softmax(logits, dim=-1)
            topv, topi = probs.topk(self.top_k, dim=-1)
            act_row, read_row = [], []
            for pos in range(len(ids)):
                toks = [self.tok.decode([j]).replace("\n", "\\n")
                        for j in topi[pos].tolist()]
                wts = topv[pos].tolist()
                read_row.append(Readout(tokens=toks, weights=wts))
                # workspace activity proxy: negative entropy (peaked readout =
                # a confident, verbalisable direction)
                p = probs[pos]
                ent = float(-(p * (p + 1e-12).log()).sum())
                max_ent = math.log(p.shape[0])
                act_row.append(max(0.0, 1.0 - ent / max_ent))
            activity.append(act_row)
            readouts.append(read_row)
        return Analysis(
            prompt=prompt, model=self.model_id, tokens=str_tokens,
            n_layers=n_layers, activity=activity, readouts=readouts,
            layer_regions=regions,
        )

    def _region(self, layer: int, n_layers: int) -> str:
        frac = layer / max(1, n_layers - 1)
        if frac < 0.28:
            return "sensory"
        if frac > 0.78:
            return "motor"
        return "workspace"

    # Steering / search / feature on a real model are heavier; for the mock
    # explorer we reuse the mock implementations for those panels so the UI is
    # fully functional regardless of backend.
    def steer(self, prompt: str, concept: str, strength: float) -> Dict:
        from .mock import MockBackend
        return MockBackend(self.n_layers, self.top_k).steer(prompt, concept, strength)

    def search(self, query: str) -> List[Dict]:
        from .mock import MockBackend
        return MockBackend(self.n_layers, self.top_k).search(query)

    def feature(self, concept: str) -> Dict:
        from .mock import MockBackend
        return MockBackend(self.n_layers, self.top_k).feature(concept)
