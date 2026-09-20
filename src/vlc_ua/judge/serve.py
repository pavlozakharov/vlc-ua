"""Minimal HTTP server speaking the System One contract over a local judge.

``POST /v1/systemone`` with ``{"state": ..., "questions": {...}}`` returns
``{"model": ..., "answers": {...}, "usage": {...}}``. Any client written for
the vendor (including the official SDK pointed at this base URL) works
against a local head. Standard library only; one request at a time, which
is what a CPU box wants anyway.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Mapping

from .backend import Judge
from .types import question_from_dict


def make_handler(judge: Judge, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in ("/v1/systemone", "/v1/ask"):
                self._send(404, {"error": "unknown path"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                questions = {k: question_from_dict(v) for k, v in body["questions"].items()}
                answers = judge.ask(body["state"], questions)
            except Exception as exc:
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send(200, {"model": model_name,
                             "answers": {k: a.as_dict() for k, a in answers.items()},
                             "usage": {"input_tokens": 0, "output_tokens": 0}})

        def _send(self, code: int, payload: Mapping) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a) -> None:  # quiet
            pass

    return Handler


def serve(judge: Judge, host: str = "127.0.0.1", port: int = 8009, model_name: str = "vlc-judge") -> None:
    HTTPServer((host, port), make_handler(judge, model_name)).serve_forever()
