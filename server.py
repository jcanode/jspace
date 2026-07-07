#!/usr/bin/env python3
"""J-Space Workspace Explorer — zero-dependency server.

Serves the static frontend and a small JSON API backed by the J-lens engine.
Runs on the Python standard library alone (mock backend). Pass ``--model`` with
a HuggingFace id (e.g. ``gpt2``) to analyse a real open-source model when
``torch``/``transformers`` are installed.

    python3 server.py                # mock backend, http://localhost:8000
    python3 server.py --model gpt2   # real GPT-2 (needs torch+transformers)
    python3 server.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from jlens import get_backend
from jlens import knowledge as kb

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

BACKEND = None  # set in main()

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "JSpace/1.0"

    # -- utilities ---------------------------------------------------------
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _serve_static(self, path):
        if path in ("/", ""):
            path = "/index.html"
        safe = os.path.normpath(path).lstrip("/")
        full = os.path.join(FRONTEND_DIR, safe)
        if not full.startswith(FRONTEND_DIR) or not os.path.isfile(full):
            self.send_error(404, "Not found")
            return
        ext = os.path.splitext(full)[1]
        ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        if path == "/api/config":
            return self._json({
                "backend": BACKEND.name,
                "display_name": BACKEND.display_name,
                "n_layers": BACKEND.n_layers,
                "steering_concepts": [
                    {"id": k, "label": v["label"]}
                    for k, v in kb.STEERING_CONCEPTS.items()
                ],
                "example_prompts": EXAMPLE_PROMPTS,
                "example_features": EXAMPLE_FEATURES,
            })
        if path == "/api/search":
            q = (qs.get("q", [""])[0]).strip()
            return self._json({"query": q, "results": BACKEND.search(q)})
        if path == "/api/feature":
            c = (qs.get("c", [""])[0]).strip()
            return self._json(BACKEND.feature(c))
        return self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            body = self._read_json()
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "empty prompt"}, status=400)
            return self._json(BACKEND.analyze(prompt).as_dict())
        if parsed.path == "/api/steer":
            body = self._read_json()
            return self._json(BACKEND.steer(
                (body.get("prompt") or "").strip(),
                body.get("concept") or "",
                float(body.get("strength", 0.5)),
            ))
        self.send_error(404, "Not found")


EXAMPLE_PROMPTS = [
    "The capital of France is",
    "A spider walking across the floor has this many legs:",
    "She opened the letter and felt a sudden wave of",
    "To define a function in Python you write",
    "Water is made of hydrogen and",
    "The moon slowly rose over the silent ocean",
]

EXAMPLE_FEATURES = ["france", "spider", "happy", "python", "moon", "capital"]


def main():
    global BACKEND
    ap = argparse.ArgumentParser(description="J-Space Workspace Explorer server")
    ap.add_argument("--model", default="mock",
                    help="'mock' (default) or a HuggingFace model id like 'gpt2'")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--layers", type=int, default=16,
                    help="layer count for the mock backend")
    args = ap.parse_args()

    if args.model == "mock":
        BACKEND = get_backend("mock", n_layers=args.layers)
    else:
        BACKEND = get_backend(args.model)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[jspace] backend = {BACKEND.display_name} ({BACKEND.n_layers} layers)")
    print(f"[jspace] serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[jspace] bye")


if __name__ == "__main__":
    main()
