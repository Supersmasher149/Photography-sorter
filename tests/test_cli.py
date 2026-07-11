from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_top_level_main_module_imports(self):
        import importlib.util

        main_path = ROOT / "main.py"
        spec = importlib.util.spec_from_file_location("photo_cull_main_test", main_path)
        module = importlib.util.module_from_spec(spec)

        assert spec.loader is not None
        spec.loader.exec_module(module)

    def test_cli_module_imports(self):
        parser = build_parser()
        self.assertEqual(parser.prog, "photo-cull-ai")

    def test_parser_defaults_match_spec(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "in", "--output", "out.jsonl"])

        self.assertEqual(args.input, "in")
        self.assertEqual(args.output, "out.jsonl")
        self.assertEqual(args.model, "qwen2.5vl:3b")
        self.assertEqual(args.max_size, 2048)
        self.assertFalse(args.recursive)
        self.assertFalse(args.resume)

    @mock.patch("photo_cull_ai.cli.run_batch")
    @mock.patch("photo_cull_ai.cli.validate_startup")
    def test_main_propagates_all_arguments(self, validate_startup, run_batch):
        result = main(
            [
                "--input",
                "photos",
                "--output",
                "results.jsonl",
                "--model",
                "vision-model",
                "--max-size",
                "1200",
                "--recursive",
                "--resume",
            ]
        )

        self.assertEqual(result, 0)
        validate_startup.assert_called_once_with(
            input_dir=Path("photos"),
            output_path=Path("results.jsonl"),
            model="vision-model",
            max_size=1200,
        )
        run_batch.assert_called_once_with(
            input_dir=Path("photos"),
            output_path=Path("results.jsonl"),
            model="vision-model",
            max_size=1200,
            recursive=True,
            resume=True,
        )

    @mock.patch("photo_cull_ai.cli.validate_startup", side_effect=RuntimeError("offline"))
    def test_main_reports_startup_failure_and_exits_two(self, _validate_startup):
        stderr = StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
            main(["--input", "photos", "--output", "results.jsonl"])

        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(stderr.getvalue(), "Error: offline\n")
