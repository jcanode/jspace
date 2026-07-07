"""Deterministic mock J-lens backend.

Produces synthetic but *structured* Jacobian-lens data that mirrors the
qualitative findings of the Global Workspace paper:

  * Early layers behave like a "sensory" region: readouts echo the input token
    itself and its orthographic sub-pieces; workspace activity is low.
  * Middle layers form the "workspace": readouts surface semantically related
    concepts (from the knowledge base) and activity peaks — especially on
    content words.
  * Late layers act like a "motor" region: readouts collapse onto a likely
    next token / continuation.

Everything is seeded off the prompt so results are stable across requests.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .base import Analysis, Backend, Readout
from . import knowledge as kb


class MockBackend(Backend):
    name = "mock"
    display_name = "Mock J-Lens (synthetic)"

    def __init__(self, n_layers: int = 16, top_k: int = 6):
        self.n_layers = n_layers
        self.top_k = top_k

    # -- helpers -----------------------------------------------------------

    def tokenize(self, text: str) -> List[str]:
        return kb.tokenize(text)

    def _region(self, layer: int) -> str:
        frac = layer / max(1, self.n_layers - 1)
        if frac < 0.28:
            return "sensory"
        if frac > 0.78:
            return "motor"
        return "workspace"

    def _activity(self, layer: int, token: str, pos: int, n_pos: int) -> float:
        """Workspace engagement bell-curve over depth, gated by content."""
        frac = layer / max(1, self.n_layers - 1)
        # bell centred at ~0.55 depth
        bell = math.exp(-((frac - 0.55) ** 2) / (2 * 0.16 ** 2))
        content = 1.0 if kb.is_content_word(token) else 0.28
        known = 1.0 if kb.related_concepts(token) else 0.6
        # a little position-dependent jitter, deterministic
        jitter = (kb.stable_hash(token, layer, pos) % 1000) / 1000.0
        val = bell * content * known * (0.82 + 0.18 * jitter)
        # content words late in the prompt get a small recency bump (they're
        # what the model is actively "thinking about")
        val *= 0.9 + 0.1 * (pos / max(1, n_pos - 1))
        return max(0.0, min(1.0, val))

    def _next_token_guess(self, tokens: List[str]) -> List[tuple]:
        """Pick a plausible continuation from the last known content word."""
        for tok in reversed(tokens):
            rel = kb.related_concepts(tok)
            if rel:
                return rel
        # fall back to generic continuation
        return [(t, 1.0 - i * 0.1) for i, t in enumerate(kb.FILLER[:4])]

    def _readout(self, layer: int, pos: int, tokens: List[str]) -> Readout:
        token = tokens[pos]
        region = self._region(layer)
        pairs: List[tuple] = []

        if region == "sensory":
            for i, v in enumerate(kb.subword_variants(token)[: self.top_k]):
                pairs.append((v, 1.0 - i * 0.12))
        elif region == "workspace":
            rel = kb.related_concepts(token)
            if rel:
                pairs.extend(rel)
            # blend in a bit of the token itself and a neighbour's concept
            pairs.append((token, 0.35))
            if pos > 0:
                nb = kb.related_concepts(tokens[pos - 1])
                if nb:
                    pairs.append((nb[0][0], 0.3))
            if not rel:
                # unknown content word: surface generic "abstract" placeholders
                for i, v in enumerate(kb.seeded_pick(
                        ["concept", "meaning", "idea", "topic", token], self.top_k,
                        token, layer, pos)):
                    pairs.append((v, 0.6 - i * 0.08))
        else:  # motor
            guess = self._next_token_guess(tokens[: pos + 1])
            pairs.extend(guess)
            pairs.append((token, 0.25))

        # add deterministic filler so every cell has top_k entries
        while len(pairs) < self.top_k:
            fill = kb.seeded_pick(kb.FILLER, self.top_k - len(pairs) + 2,
                                  token, layer, pos, len(pairs))
            for f in fill:
                if f not in [p[0] for p in pairs]:
                    pairs.append((f, 0.15))
                if len(pairs) >= self.top_k:
                    break
            break

        # dedupe by token keeping max weight, sort, normalise
        best: Dict[str, float] = {}
        for t, w in pairs:
            best[t] = max(best.get(t, 0.0), float(w))
        items = sorted(best.items(), key=lambda kv: -kv[1])[: self.top_k]
        total = sum(w for _, w in items) or 1.0
        toks = [t for t, _ in items]
        wts = [w / total for _, w in items]
        return Readout(tokens=toks, weights=wts)

    # -- public API --------------------------------------------------------

    def analyze(self, prompt: str) -> Analysis:
        tokens = self.tokenize(prompt)
        n_pos = len(tokens)
        activity: List[List[float]] = []
        readouts: List[List[Readout]] = []
        regions: List[str] = []
        for layer in range(self.n_layers):
            regions.append(self._region(layer))
            act_row, read_row = [], []
            for pos in range(n_pos):
                act_row.append(self._activity(layer, tokens[pos], pos, n_pos))
                read_row.append(self._readout(layer, pos, tokens))
            activity.append(act_row)
            readouts.append(read_row)
        return Analysis(
            prompt=prompt, model=self.name, tokens=tokens,
            n_layers=self.n_layers, activity=activity,
            readouts=readouts, layer_regions=regions,
        )

    def steer(self, prompt: str, concept: str, strength: float) -> Dict:
        spec = kb.STEERING_CONCEPTS.get(concept)
        tokens = self.tokenize(prompt)
        base_next = [t for t, _ in self._next_token_guess(tokens)][:5]
        if not spec:
            return {"concept": concept, "baseline": base_next,
                    "steered": base_next, "note": "unknown concept"}
        amp = spec["tokens"]
        # strength 0..1 mixes steering tokens into the continuation
        n_from_concept = int(round(strength * min(len(amp), 4)))
        steered = amp[:n_from_concept] + [t for t in base_next
                                          if t not in amp[:n_from_concept]]
        steered = steered[:5]
        # a mock generated sentence
        cont = " ".join(steered[:3])
        return {
            "concept": concept,
            "label": spec["label"],
            "strength": strength,
            "baseline": base_next,
            "steered": steered,
            "generation": f"{prompt.rstrip()} {cont} ...",
        }

    def search(self, query: str) -> List[Dict]:
        """Find corpus sentences + concepts whose workspace surfaces `query`."""
        q = kb.normalize(query)
        results: List[Dict] = []
        # 1) direct concept matches
        for concept, rels in kb.CONCEPTS.items():
            rel_tokens = [concept] + [t.lower() for t, _ in rels]
            if q in rel_tokens or any(q in rt for rt in rel_tokens):
                results.append({
                    "type": "concept",
                    "id": concept,
                    "title": concept,
                    "related": [t for t, _ in rels][:5],
                    "score": round(1.0 if q == concept else 0.7, 3),
                })
        # 2) corpus sentences containing the query (workspace activation demo)
        for i, sent in enumerate(kb.CORPUS):
            low = sent.lower()
            if q and q in low:
                # find the matching token position
                toks = self.tokenize(sent)
                pos = next((j for j, t in enumerate(toks)
                            if kb.normalize(t) == q or q in t.lower()), 0)
                results.append({
                    "type": "activation",
                    "id": f"corpus-{i}",
                    "title": sent,
                    "position": pos,
                    "tokens": toks,
                    "peak_layer": int(self.n_layers * 0.55),
                    "score": round(0.9 - 0.02 * i, 3),
                })
        results.sort(key=lambda r: -r["score"])
        return results[:20]

    def feature(self, concept: str) -> Dict:
        """Neuronpedia-style dashboard for a concept 'feature'."""
        key = kb.normalize(concept)
        rels = kb.CONCEPTS.get(key, [])
        # top activating corpus examples: sentences mentioning the concept or its
        # relations, with the activating token highlighted.
        targets = {key} | {t.lower() for t, _ in rels}
        examples = []
        for sent in kb.CORPUS:
            toks = self.tokenize(sent)
            acts = []
            hit = False
            for t in toks:
                n = kb.normalize(t)
                a = 1.0 if n == key else (0.7 if n in targets else 0.0)
                if a > 0:
                    hit = True
                acts.append(round(a, 3))
            if hit:
                examples.append({"tokens": toks, "acts": acts,
                                 "max_act": max(acts)})
        examples.sort(key=lambda e: -e["max_act"])
        # activation histogram (mock, right-skewed)
        hist = [0.42, 0.24, 0.14, 0.09, 0.05, 0.03, 0.02, 0.01]
        return {
            "id": key,
            "label": concept,
            "region": "workspace",
            "peak_layer": int(self.n_layers * 0.55),
            "related": [{"token": t, "weight": round(w, 3)} for t, w in rels],
            "examples": examples[:8],
            "histogram": hist,
            "density": round(len(examples) / max(1, len(kb.CORPUS)), 3),
        }
