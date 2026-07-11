from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.core import ensure_ollama_ready, grade_with_ollama


class OllamaTests(unittest.TestCase):
    def test_grade_with_ollama_sends_expected_request_and_reads_dict_response(self):
        class Client:
            def __init__(self):
                self.request = None

            def chat(self, **kwargs):
                self.request = kwargs
                return {"message": {"content": '{"rating": 4}'}}

        client = Client()
        image_path = Path("preview.jpg")

        response = grade_with_ollama(image_path, "vision-model", client=client)

        self.assertEqual(response, '{"rating": 4}')
        self.assertEqual(client.request["model"], "vision-model")
        self.assertEqual(client.request["messages"][0]["images"], ["preview.jpg"])
        self.assertEqual(client.request["options"]["temperature"], 0)

    def test_grade_with_ollama_reads_object_response(self):
        client = SimpleNamespace(
            chat=lambda **_kwargs: SimpleNamespace(
                message=SimpleNamespace(content="object response")
            )
        )

        self.assertEqual(
            grade_with_ollama(Path("preview.jpg"), "vision-model", client=client),
            "object response",
        )

    def test_grade_with_ollama_rejects_missing_content(self):
        client = SimpleNamespace(chat=lambda **_kwargs: {"message": {}})

        with self.assertRaises(ValueError) as exc:
            grade_with_ollama(Path("preview.jpg"), "vision-model", client=client)

        self.assertIn("message content", str(exc.exception))

    def test_ensure_ollama_ready_accepts_dict_and_object_model_entries(self):
        listing = {
            "models": [
                {"model": "first:latest"},
                SimpleNamespace(name="vision-model"),
            ]
        }
        client = SimpleNamespace(list=lambda: listing)

        ensure_ollama_ready("vision-model", client=client)

    def test_ensure_ollama_ready_rejects_missing_model(self):
        client = SimpleNamespace(list=lambda: {"models": [{"name": "other"}]})

        with self.assertRaises(RuntimeError) as exc:
            ensure_ollama_ready("vision-model", client=client)

        self.assertIn("vision-model", str(exc.exception))

    def test_ensure_ollama_ready_wraps_connection_failure(self):
        def fail():
            raise ConnectionError("offline")

        client = SimpleNamespace(list=fail)

        with self.assertRaises(RuntimeError) as exc:
            ensure_ollama_ready("vision-model", client=client)

        self.assertIn("Unable to contact", str(exc.exception))
        self.assertIsInstance(exc.exception.__cause__, ConnectionError)
