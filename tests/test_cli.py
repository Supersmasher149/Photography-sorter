from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.cli import build_parser


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
