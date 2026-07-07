"""Concept knowledge base and token utilities for the mock J-lens.

The mock backend needs to produce *plausible* Jacobian-lens readouts: given a
prompt, the intermediate ("workspace") layers should surface concepts that are
semantically related to the tokens in the prompt, while early layers surface the
input tokens themselves and late layers surface a likely next token.

This module holds a small hand-curated association graph plus deterministic
token-mangling helpers so that arbitrary prompts still yield believable output.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Association graph: token -> related "verbalizable" concepts.
#
# These stand in for the tokens a real J-lens would rank highly in the model's
# workspace when a given concept is active. Weights are relative and get
# normalised at read time.
# ---------------------------------------------------------------------------

CONCEPTS: Dict[str, List[Tuple[str, float]]] = {
    # Geography
    "france": [("Paris", 1.0), ("Europe", 0.7), ("capital", 0.6), ("French", 0.6), ("wine", 0.4)],
    "paris": [("France", 1.0), ("Eiffel", 0.6), ("capital", 0.6), ("city", 0.5), ("Seine", 0.4)],
    "japan": [("Tokyo", 1.0), ("Asia", 0.7), ("island", 0.5), ("Japanese", 0.6), ("sushi", 0.4)],
    "tokyo": [("Japan", 1.0), ("capital", 0.6), ("city", 0.5), ("Asia", 0.5)],
    "italy": [("Rome", 1.0), ("Europe", 0.7), ("pasta", 0.5), ("Italian", 0.6), ("pizza", 0.4)],
    "germany": [("Berlin", 1.0), ("Europe", 0.7), ("German", 0.6), ("capital", 0.6)],
    "spain": [("Madrid", 1.0), ("Europe", 0.7), ("Spanish", 0.6), ("capital", 0.6)],
    "egypt": [("Cairo", 1.0), ("pyramid", 0.7), ("Nile", 0.6), ("Africa", 0.6), ("desert", 0.5)],
    "china": [("Beijing", 1.0), ("Asia", 0.7), ("Chinese", 0.6), ("Great", 0.4)],
    "london": [("England", 1.0), ("capital", 0.6), ("Thames", 0.5), ("British", 0.5)],
    "capital": [("city", 1.0), ("country", 0.7), ("government", 0.5), ("Paris", 0.4)],
    # Animals
    "spider": [("legs", 1.0), ("eight", 0.9), ("web", 0.7), ("insect", 0.5), ("arachnid", 0.6)],
    "dog": [("animal", 1.0), ("bark", 0.7), ("pet", 0.7), ("loyal", 0.5), ("puppy", 0.5)],
    "cat": [("animal", 1.0), ("meow", 0.7), ("pet", 0.7), ("feline", 0.6), ("whiskers", 0.4)],
    "bird": [("fly", 1.0), ("feathers", 0.7), ("wings", 0.7), ("nest", 0.5), ("sing", 0.4)],
    "fish": [("water", 1.0), ("swim", 0.8), ("gills", 0.6), ("ocean", 0.5), ("scales", 0.4)],
    "elephant": [("large", 1.0), ("trunk", 0.8), ("Africa", 0.5), ("gray", 0.5), ("tusks", 0.5)],
    "octopus": [("eight", 1.0), ("arms", 0.8), ("ocean", 0.6), ("ink", 0.5), ("tentacle", 0.6)],
    # Science / math
    "water": [("H2O", 1.0), ("liquid", 0.7), ("wet", 0.6), ("ocean", 0.5), ("drink", 0.4)],
    "sun": [("star", 1.0), ("light", 0.8), ("hot", 0.6), ("solar", 0.6), ("yellow", 0.4)],
    "moon": [("orbit", 1.0), ("night", 0.7), ("Earth", 0.6), ("crater", 0.5), ("lunar", 0.6)],
    "gravity": [("force", 1.0), ("mass", 0.7), ("Newton", 0.6), ("fall", 0.6), ("Einstein", 0.4)],
    "atom": [("nucleus", 1.0), ("electron", 0.8), ("proton", 0.7), ("small", 0.5), ("particle", 0.6)],
    "two": [("2", 1.0), ("pair", 0.7), ("three", 0.5), ("number", 0.5), ("even", 0.4)],
    "three": [("3", 1.0), ("triangle", 0.6), ("number", 0.5), ("four", 0.5), ("trio", 0.4)],
    "eight": [("8", 1.0), ("number", 0.6), ("octet", 0.5), ("legs", 0.4)],
    "plus": [("add", 1.0), ("sum", 0.7), ("equals", 0.6), ("math", 0.5)],
    "equals": [("result", 1.0), ("answer", 0.7), ("sum", 0.6), ("is", 0.5)],
    # Emotions / abstract
    "happy": [("joy", 1.0), ("smile", 0.7), ("glad", 0.6), ("positive", 0.5), ("emotion", 0.5)],
    "sad": [("sorrow", 1.0), ("cry", 0.7), ("unhappy", 0.6), ("emotion", 0.5), ("tears", 0.4)],
    "love": [("affection", 1.0), ("heart", 0.7), ("care", 0.6), ("emotion", 0.5), ("romance", 0.4)],
    "fear": [("afraid", 1.0), ("danger", 0.7), ("scared", 0.6), ("emotion", 0.5)],
    # Programming
    "python": [("code", 1.0), ("snake", 0.5), ("programming", 0.8), ("script", 0.6), ("language", 0.5)],
    "function": [("return", 1.0), ("call", 0.7), ("argument", 0.6), ("code", 0.5), ("def", 0.5)],
    "variable": [("value", 1.0), ("assign", 0.7), ("name", 0.6), ("store", 0.5)],
    "model": [("network", 1.0), ("train", 0.7), ("weights", 0.6), ("neural", 0.6), ("predict", 0.5)],
    # Food
    "apple": [("fruit", 1.0), ("red", 0.6), ("tree", 0.5), ("eat", 0.5), ("pie", 0.4)],
    "bread": [("bake", 1.0), ("flour", 0.7), ("food", 0.6), ("wheat", 0.5), ("loaf", 0.4)],
    "coffee": [("caffeine", 1.0), ("drink", 0.7), ("morning", 0.6), ("bean", 0.5), ("hot", 0.4)],
    # Colors
    "red": [("color", 1.0), ("blood", 0.5), ("fire", 0.5), ("stop", 0.4), ("rose", 0.4)],
    "blue": [("color", 1.0), ("sky", 0.6), ("ocean", 0.5), ("sad", 0.3), ("cold", 0.4)],
    "green": [("color", 1.0), ("grass", 0.6), ("nature", 0.5), ("go", 0.4), ("leaf", 0.4)],
}

# Concepts that a user can "steer" toward in the Steering panel, with a short
# label and the tokens that steering amplifies.
STEERING_CONCEPTS: Dict[str, Dict] = {
    "geography": {
        "label": "Geography / capitals",
        "tokens": ["Paris", "capital", "country", "city", "Europe", "map"],
    },
    "arachnids": {
        "label": "Spiders / eight legs",
        "tokens": ["legs", "eight", "web", "spider", "arachnid"],
    },
    "emotion": {
        "label": "Emotional tone",
        "tokens": ["feel", "joy", "sad", "heart", "emotion", "love"],
    },
    "code": {
        "label": "Programming",
        "tokens": ["def", "return", "function", "code", "variable", "loop"],
    },
    "formality": {
        "label": "Formal register",
        "tokens": ["therefore", "hereby", "regarding", "furthermore", "shall"],
    },
    "poetry": {
        "label": "Poetic / lyrical",
        "tokens": ["whisper", "shadow", "golden", "silent", "dream", "moon"],
    },
}

# A tiny background corpus used to build feature dashboards (top activating
# examples) in the Neuronpedia-style feature view.
CORPUS = [
    "The capital of France is Paris and it sits on the Seine.",
    "A spider has eight legs and spins a silken web.",
    "The octopus stretched its eight arms through the dark water.",
    "She felt a wave of joy as the sun rose over the ocean.",
    "To define a function in Python you write def followed by a name.",
    "The moon orbits the Earth once every twenty seven days.",
    "He was afraid of the shadow moving in the silent hallway.",
    "Tokyo is the bustling capital city of Japan.",
    "Water is made of two hydrogen atoms and one oxygen atom.",
    "The dog barked loudly at the passing mail truck.",
    "Gravity is the force that pulls objects toward mass.",
    "A neural model learns weights that predict the next token.",
    "The red apple fell from the tree onto the green grass.",
    "Coffee gives a warm jolt of caffeine every single morning.",
    "Rome, the capital of Italy, is famous for pizza and pasta.",
]

# Common English filler tokens for early/late layers when nothing better fits.
FILLER = [
    "the", "a", "of", "and", "to", "in", "is", "that", "it", "for",
    "on", "with", "as", "was", "at", "by", "an", "be", "this", "which",
    "or", "from", "but", "not", "are", "we", "they", "he", "she", "you",
]

_word_re = re.compile(r"[A-Za-z0-9']+|[^A-Za-z0-9\s]")


def tokenize(text: str) -> List[str]:
    """Very small word/punctuation tokenizer (stands in for a BPE tokenizer)."""
    toks = _word_re.findall(text.strip())
    return toks if toks else ["<empty>"]


def normalize(token: str) -> str:
    t = token.lower().strip("'\".,!?;:()[]")
    # naive singularisation so "spiders" hits "spider"
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("es") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def is_content_word(token: str) -> bool:
    """Rough heuristic: content words carry the workspace signal."""
    t = token.lower().strip("'\".,!?;:()[]")
    if not t or not t[0].isalnum():
        return False
    return t not in set(FILLER) and len(t) > 2 or t.isdigit()


def related_concepts(token: str) -> List[Tuple[str, float]]:
    """Return (concept_token, weight) associations for a token, if known."""
    key = normalize(token)
    if key in CONCEPTS:
        return CONCEPTS[key]
    # capitalised proper-noun-ish token we don't know: relate to itself + generics
    return []


def subword_variants(token: str) -> List[str]:
    """Orthographic neighbours a low ('sensory') layer might surface."""
    t = token.strip()
    out = [t]
    if len(t) > 3:
        out.append(t[: max(2, len(t) // 2)])  # prefix piece
        out.append(t[-max(2, len(t) // 2):])  # suffix piece
    if t and t[0].isalpha():
        out.append(" " + t)  # leading-space BPE variant
        out.append(t.lower() if t[0].isupper() else t.capitalize())
    # dedupe preserving order
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def stable_hash(*parts) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16)


def seeded_pick(pool: List[str], n: int, *seed_parts) -> List[str]:
    """Deterministically pick n items from pool based on a seed."""
    if not pool:
        return []
    h = stable_hash(*seed_parts)
    res, i, used = [], 0, set()
    while len(res) < n and len(used) < len(pool):
        idx = (h + i * 2654435761) % len(pool)
        if idx not in used:
            used.add(idx)
            res.append(pool[idx])
        i += 1
    return res
