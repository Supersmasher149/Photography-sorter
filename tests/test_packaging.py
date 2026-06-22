from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_pyproject_defines_console_script(self):
        pyproject = ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

        self.assertEqual(data["project"]["name"], "photo-cull-ai")
        self.assertEqual(
            data["project"]["scripts"]["photo-cull-ai"],
            "photo_cull_ai.cli:main",
        )
