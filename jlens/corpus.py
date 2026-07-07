"""Corpus backend: a tiny n-gram language model, learned offline at startup.

Unlike the synthetic ``MockBackend`` (whose readouts come from a hand-curated
concept graph), this backend *learns* its statistics from a bundled text corpus
when the server boots — no downloads, no `torch`, no `transformers`, pure Python
standard library. It's a genuine (if very small) statistical language model.

It reuses the mock backend's analysis loop and its steer/search/feature panels,
overriding only how each (layer, position) cell is filled so that:

  * sensory layers  -> echo the input token (orthographic sub-pieces)
  * workspace layers -> distributionally *related* tokens, ranked by pointwise
                        mutual information from the corpus co-occurrence stats
  * motor layers    -> the learned next-token distribution (bigram / trigram)

Where the corpus is too sparse to say anything, it falls back to the mock
backend's readout so every cell is still meaningful.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .base import Readout
from .mock import MockBackend
from . import knowledge as kb


class CorpusBackend(MockBackend):
    name = "corpus"

    def __init__(self, n_layers: int = 16, top_k: int = 6, window: int = 3):
        super().__init__(n_layers=n_layers, top_k=top_k)
        self.display_name = "Corpus n-gram (learned offline)"
        self.window = window
        self._train(kb.CORPUS)

    @staticmethod
    def _norm(t: str) -> str:
        # keep readable word forms (don't singularise) for corpus tokens
        return kb.normalize(t, singularize=False)

    # -- training ----------------------------------------------------------

    def _train(self, corpus: List[str]) -> None:
        self.unigram: Counter = Counter()
        self.bigram: Dict[str, Counter] = defaultdict(Counter)      # w -> next
        self.trigram: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.cooc: Dict[str, Counter] = defaultdict(Counter)        # window co-occ
        self.total = 0
        stop = set(kb.FILLER)
        for line in corpus:
            toks = [self._norm(t) for t in self.tokenize(line)]
            toks = [t for t in toks if t and t[0].isalnum()]
            self.total += len(toks)
            self.unigram.update(toks)
            for i, w in enumerate(toks):
                if i + 1 < len(toks):
                    self.bigram[w][toks[i + 1]] += 1
                if i + 2 < len(toks):
                    self.trigram[(w, toks[i + 1])][toks[i + 2]] += 1
                for j in range(max(0, i - self.window), min(len(toks), i + self.window + 1)):
                    if j != i and toks[j] not in stop:
                        self.cooc[w][toks[j]] += 1
        self.vocab = set(self.unigram)

    # -- learned readouts --------------------------------------------------

    def _pmi_related(self, token: str) -> List[Tuple[str, float]]:
        """Tokens distributionally related to `token`, ranked by PMI."""
        key = self._norm(token)
        co = self.cooc.get(key)
        if not co:
            return []
        out = []
        cw = self.unigram[key] or 1
        for c, n in co.items():
            if c == key:
                continue
            # pmi = log( p(c|w) / p(c) ) = log( (n/cw) / (count_c/total) )
            pc = self.unigram[c] / max(1, self.total)
            pmi = math.log(((n / cw) + 1e-9) / (pc + 1e-9))
            out.append((c, pmi * math.log(1 + n)))  # weight by evidence
        out.sort(key=lambda kv: -kv[1])
        # renormalise scores to positive-ish weights
        return [(c, max(0.05, s)) for c, s in out[: self.top_k]]

    def _next_dist(self, tokens: List[str], pos: int) -> List[Tuple[str, float]]:
        """Learned next-token distribution given context ending at `pos`."""
        cur = self._norm(tokens[pos])
        prev = self._norm(tokens[pos - 1]) if pos > 0 else None
        dist = None
        if prev is not None and (prev, cur) in self.trigram:
            dist = self.trigram[(prev, cur)]
        elif cur in self.bigram:
            dist = self.bigram[cur]
        if not dist:
            return []
        items = dist.most_common(self.top_k)
        total = sum(c for _, c in items) or 1
        return [(w, c / total) for w, c in items]

    def _confidence(self, pairs: List[Tuple[str, float]]) -> float:
        """Peakedness of a distribution -> workspace-activity signal (0..1)."""
        if not pairs:
            return 0.0
        ws = [max(0.0, w) for _, w in pairs]
        s = sum(ws) or 1.0
        p = [w / s for w in ws]
        ent = -sum(x * math.log(x + 1e-12) for x in p if x > 0)
        max_ent = math.log(len(p)) or 1.0
        return max(0.0, 1.0 - ent / max_ent)

    # -- overrides used by MockBackend.analyze() ---------------------------

    def _readout(self, layer: int, pos: int, tokens: List[str]) -> Readout:
        region = self._region(layer)
        pairs: List[Tuple[str, float]] = []
        if region == "workspace":
            pairs = self._pmi_related(tokens[pos])
        elif region == "motor":
            pairs = self._next_dist(tokens, pos)
        # sensory region, or any empty learned readout, falls back to the mock's
        # deterministic behaviour so the cell is never blank.
        if not pairs:
            return super()._readout(layer, pos, tokens)
        # normalise + pad
        best: Dict[str, float] = {}
        for t, w in pairs:
            best[t] = max(best.get(t, 0.0), float(w))
        items = sorted(best.items(), key=lambda kv: -kv[1])[: self.top_k]
        total = sum(w for _, w in items) or 1.0
        return Readout(tokens=[t for t, _ in items],
                       weights=[w / total for _, w in items])

    def _activity(self, layer: int, token: str, pos: int, n_pos: int) -> float:
        # depth bell curve (sensory->workspace->motor), same as mock ...
        frac = layer / max(1, self.n_layers - 1)
        bell = math.exp(-((frac - 0.55) ** 2) / (2 * 0.16 ** 2))
        # ... modulated by how confident the *learned* distribution is here.
        region = self._region(layer)
        if region == "workspace":
            conf = self._confidence(self._pmi_related(token))
        elif region == "motor":
            conf = self._confidence(self._next_dist([token], 0)
                                    or self._pmi_related(token))
        else:
            conf = 0.25
        content = 1.0 if kb.is_content_word(token) else 0.3
        return max(0.0, min(1.0, bell * content * (0.35 + 0.65 * conf)))
