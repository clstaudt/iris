"""A scripted language model behind Ollama's API, for testing the voice.

It answers at once and always the same way, so a test measures Iris rather
than a model: a short reply to a question, and a long one when asked for a
story or a long answer. How long it takes to start and how fast it writes
are set, so a test can stand in for a fast or a slow model.
"""

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "voicetest:latest"

SHORT = "Paris is the capital of France. It lies on the Seine."
LONG = " ".join(
    f"This is sentence number {n} of a long answer, which goes on for a while."
    for n in range(1, 9)
)
# A request for a long reply, so the voice has time to be interrupted
LONG_REQUEST = re.compile(r"\b(story|long|everything|in detail)\b", re.IGNORECASE)


def reply_to(messages: list[dict]) -> str:
    """The reply to the conversation: long when the last user message asks
    for it, short otherwise."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return LONG if LONG_REQUEST.search(message.get("content") or "") else SHORT
    return SHORT


class StubModel:
    """Ollama's API on a local port, answering with scripted replies."""

    def __init__(self, first_token: float = 0.3, per_word: float = 0.02):
        """
        Args:
            first_token: Seconds before the reply's first word
            per_word: Seconds per further word
        """
        self.first_token = first_token
        self.per_word = per_word
        # The requests the model was sent, newest last
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _json(self, body: dict, status: int = 200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> dict:
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length) or b"{}")

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                if self.path == "/api/version":
                    self._json({"version": "0.0.0-voicetest"})
                elif self.path == "/api/tags":
                    self._json({"models": [stub._entry()]})
                elif self.path == "/":
                    self._json({"status": "ok"})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                body = self._body()
                if self.path == "/api/show":
                    self._json(stub._show())
                elif self.path == "/api/chat":
                    stub.requests.append(body)
                    stub._chat(self, body)
                else:
                    self._json({"error": "not found"}, 404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self):
        self._server.shutdown()

    @staticmethod
    def _details() -> dict:
        return {
            "format": "gguf",
            "family": "voicetest",
            "families": ["voicetest"],
            "parameter_size": "1B",
            "quantization_level": "Q4_0",
        }

    def _entry(self) -> dict:
        return {
            "name": MODEL,
            "model": MODEL,
            "modified_at": "2026-01-01T00:00:00Z",
            "size": 1,
            "digest": "0" * 64,
            "details": self._details(),
        }

    def _show(self) -> dict:
        return {
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": self._details(),
            "model_info": {"general.architecture": "voicetest", "voicetest.context_length": 32768},
            "capabilities": ["completion"],
        }

    def _chat(self, handler: BaseHTTPRequestHandler, body: dict):
        text = reply_to(body.get("messages") or [])
        words = text.split(" ")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-ndjson")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()

        def send(payload: dict):
            data = (json.dumps(payload) + "\n").encode()
            handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            handler.wfile.flush()

        def chunk(content: str, done: bool) -> dict:
            payload = {
                "model": MODEL,
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": content},
                "done": done,
            }
            if done:
                payload["done_reason"] = "stop"
            return payload

        try:
            time.sleep(self.first_token)
            for n, word in enumerate(words):
                if n:
                    time.sleep(self.per_word)
                send(chunk(word if n == 0 else " " + word, False))
            send(chunk("", True))
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Iris cut the request off (Stop, or the user went on)
            pass
