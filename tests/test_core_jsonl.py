from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.core import (
    append_jsonl,
    build_error_record,
    load_completed_paths,
    parse_model_json,
)


class CoreJsonlTests(unittest.TestCase):
    def test_append_jsonl_writes_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ratings.jsonl"

            append_jsonl(output_path, {"raw_path": "/a.CR3"})
            append_jsonl(output_path, {"raw_path": "/b.CR3"})

            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), {"raw_path": "/a.CR3"})
            self.assertEqual(json.loads(lines[1]), {"raw_path": "/b.CR3"})

    def test_load_completed_paths_reads_existing_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ratings.jsonl"
            output_path.write_text(
                '\n'.join(
                    [
                        json.dumps({"raw_path": "/photos/IMG_0001.CR3"}),
                        json.dumps({"raw_path": "/photos/IMG_0002.CR3"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_completed_paths(output_path),
                {"/photos/IMG_0001.CR3", "/photos/IMG_0002.CR3"},
            )

    def test_load_completed_paths_rejects_non_object_with_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ratings.jsonl"
            output_path.write_text("[]\n", encoding="utf-8")

            with self.assertRaises(ValueError) as exc:
                load_completed_paths(output_path)

            self.assertIn("line 1", str(exc.exception))
            self.assertIn("expected object", str(exc.exception))

    def test_parse_model_json_validates_required_fields(self):
        parsed = parse_model_json(
            '{"rating": 4, "keep": true, "reason": "Strong composition."}'
        )

        self.assertEqual(
            parsed,
            {
                "rating": 4,
                "keep": True,
                "reason": "Strong composition.",
            },
        )

    def test_parse_model_json_rejects_invalid_payload(self):
        with self.assertRaises(ValueError) as exc:
            parse_model_json('{"rating": 7, "keep": true, "reason": "bad"}')

        self.assertIn("rating", str(exc.exception))

    def test_build_error_record_preserves_empty_raw_response(self):
        record = build_error_record(
            raw_path=Path("/photos/IMG_0001.CR3"),
            model="test-model",
            error="Invalid JSON response",
            raw_response="",
        )

        self.assertIn("raw_response", record)
        self.assertEqual(record["raw_response"], "")
