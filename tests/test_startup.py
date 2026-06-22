from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.core import validate_startup


class StartupValidationTests(unittest.TestCase):
    def test_validate_startup_rejects_non_positive_max_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            output_path = input_dir / "ratings.jsonl"

            with self.assertRaises(ValueError) as exc:
                validate_startup(
                    input_dir=input_dir,
                    output_path=output_path,
                    model="test-model",
                    max_size=0,
                )

            self.assertIn("--max-size", str(exc.exception))

    def test_validate_startup_rejects_missing_input_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "missing"
            output_path = Path(tmp) / "ratings.jsonl"

            with self.assertRaises(ValueError) as exc:
                validate_startup(
                    input_dir=input_dir,
                    output_path=output_path,
                    model="test-model",
                    max_size=100,
                )

            self.assertIn("--input", str(exc.exception))

    def test_validate_startup_reports_missing_exiftool(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            output_path = input_dir / "ratings.jsonl"

            with self.assertRaises(RuntimeError) as exc:
                validate_startup(
                    input_dir=input_dir,
                    output_path=output_path,
                    model="test-model",
                    max_size=100,
                    exiftool_lookup=lambda _name: None,
                    ollama_ready=lambda _model: None,
                )

            self.assertIn("exiftool", str(exc.exception))

    def test_validate_startup_checks_requested_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            output_path = input_dir / "ratings.jsonl"

            with self.assertRaises(RuntimeError) as exc:
                validate_startup(
                    input_dir=input_dir,
                    output_path=output_path,
                    model="missing-model",
                    max_size=100,
                    exiftool_lookup=lambda _name: "exiftool",
                    ollama_ready=lambda model: (_ for _ in ()).throw(
                        RuntimeError(f"Ollama model '{model}' is not available locally")
                    ),
                )

            self.assertIn("missing-model", str(exc.exception))
