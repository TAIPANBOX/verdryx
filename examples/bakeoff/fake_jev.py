"""A free, local stand-in for the Jev API (TypeSafe AI), so `--dry-run`
exercises typryx's real `jev` backend code path (internal/backend/jev.go in
the typryx repo), including the key-file plumbing and the Authorization
header it sends, without spending anything or needing a key.

Replays the wire shape pinned in typryx's own
`internal/backend/testdata/jev_example_response.json` (read 2026-09-25):
`POST /systemone` -> `{"model": ..., "answers": {"q": {...}}, "usage": {...}}`.
Only the `noul` answer type is served (the bake-off's dataset only asks
`eval.outcome_met`, a noul template) -- see jev.go's `answerFrom` for the
other two shapes this fake does not attempt to replay.

The noul probability is derived deterministically from a hash of the
request's own `state` field, not randomly: the same case always gets the
same fake answer, so a dry run is reproducible. It is NOT meant to be an
accurate judge -- see the harness README's "what it does not prove".

Checks the `Authorization: Bearer <key>` header exactly the way jev.go
builds it (`"Bearer " + j.cfg.APIKey`, see tryOnce()) and answers 401 when
it does not match the key this fake was started with, so a wrong-key dry
run exercises the same failure path a real wrong key would.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DEFAULT_MODEL = "jev-1.13.0-fake"


def _noul_from_state(state: Any) -> float:
    """A deterministic pseudo-probability in [0, 1], derived from a SHA-256
    hash of the request's own canonical (sorted-key) JSON -- "a hash" per
    the brief, not a real judgement. Two different requests almost always
    get different values; the same request always gets the same one."""
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    as_int = int.from_bytes(digest[:4], "big")
    return as_int / 0xFFFFFFFF


class _Handler(BaseHTTPRequestHandler):
    #: Set on the class before the server starts (see serve() below), the
    #: same "class-level state for one handler class per server" shape
    #: tests/conftest.py's _ScriptedTypryxHandler uses.
    expected_key: str = ""
    model: str = DEFAULT_MODEL

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/systemone":
            self._json(404, {"error": "not_found"})
            return

        auth = self.headers.get("Authorization", "")
        if auth != f"Bearer {type(self).expected_key}":
            self._json(401, {"error": "unauthorized"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            request = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid_json"})
            return

        state = request.get("state", {})
        questions = request.get("questions") or {}
        q = questions.get("q") or {}
        q_type = q.get("type")

        if q_type != "noul":
            # The bake-off only ever asks eval.outcome_met (a noul
            # template); anything else is out of this fake's scope.
            self._json(422, {"error": "unsupported_question_type"})
            return

        noul = _noul_from_state(state)
        input_tokens = max(1, len(json.dumps(state)) // 4)
        self._json(
            200,
            {
                "model": type(self).model,
                "answers": {"q": {"type": "noul", "noul": noul}},
                "usage": {"input_tokens": input_tokens, "output_tokens": 8},
            },
        )

    def do_GET(self) -> None:
        self._json(404, {"error": "not_found"})

    def log_message(self, *_args: object) -> None:  # silence stderr request logging
        pass


def serve(host: str, port: int, key: str, model: str) -> ThreadingHTTPServer:
    handler = type("_BoundHandler", (_Handler,), {"expected_key": key, "model": model})
    server = ThreadingHTTPServer((host, port), handler)
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="0 picks a free port")
    p.add_argument("--key", required=True, help="the bearer key typryx's jev backend must send")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args(argv)

    server = serve(args.host, args.port, args.key, args.model)
    # Printed for bakeoff.sh to capture (it may have asked for port 0), and
    # flushed immediately since the caller is watching this line to know
    # the fake is ready to accept connections.
    print(f"FAKE_JEV_LISTENING host={args.host} port={server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
