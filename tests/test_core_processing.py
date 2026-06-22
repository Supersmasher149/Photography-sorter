from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.core import (
    PreviewExtractionError,
    extract_embedded_jpeg,
    find_raw_files,
    process_raw_file,
    run_batch,
)


class CoreProcessingTests(unittest.TestCase):
    def test_find_raw_files_returns_sorted_absolute_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "b.CR3").write_text("", encoding="utf-8")
            (tmp_path / "a.nef").write_text("", encoding="utf-8")
            (tmp_path / "notes.txt").write_text("", encoding="utf-8")
            nested = tmp_path / "nested"
            nested.mkdir()
            (nested / "c.ARW").write_text("", encoding="utf-8")

            files = find_raw_files(tmp_path, recursive=False)

            self.assertEqual(
                files,
                [
                    (tmp_path / "a.nef").resolve(),
                    (tmp_path / "b.CR3").resolve(),
                ],
            )

    def test_process_file_records_raw_response_for_invalid_model_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "IMG_0001.CR3"
            raw_path.write_text("", encoding="utf-8")
            output_path = tmp_path / "ratings.jsonl"
            preview_path = tmp_path / "preview.jpg"
            resized_path = tmp_path / "resized.jpg"
            preview_path.write_text("preview", encoding="utf-8")
            resized_path.write_text("resized", encoding="utf-8")

            record = process_raw_file(
                raw_path=raw_path,
                output_path=output_path,
                model="test-model",
                max_size=2048,
                extract_preview=lambda _: preview_path,
                resize_preview=lambda _path, _size: resized_path,
                grade_image=lambda _path, _model: "not-json",
            )

            self.assertIsNone(record["rating"])
            self.assertEqual(record["raw_response"], "not-json")
            self.assertIn("JSON", record["error"])
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0]), record)
            self.assertFalse(preview_path.exists())
            self.assertFalse(resized_path.exists())

    def test_extract_embedded_jpeg_surfaces_exiftool_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "IMG_0001.CR3"
            raw_path.write_text("", encoding="utf-8")

            def fake_run(_args, **_kwargs):
                return subprocess.CompletedProcess(
                    args=[],
                    returncode=2,
                    stdout=b"",
                    stderr=b"File format error",
                )

            with self.assertRaises(PreviewExtractionError) as exc:
                extract_embedded_jpeg(raw_path, run_command=fake_run)

            self.assertIn("File format error", str(exc.exception))

    def test_run_batch_skips_completed_paths_when_resume_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "IMG_0001.CR3"
            second = tmp_path / "IMG_0002.CR3"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")
            output_path = tmp_path / "ratings.jsonl"
            output_path.write_text(
                json.dumps({"raw_path": str(first.resolve())}) + "\n",
                encoding="utf-8",
            )
            seen: list[Path] = []

            summary = run_batch(
                input_dir=tmp_path,
                output_path=output_path,
                model="test-model",
                max_size=2048,
                recursive=False,
                resume=True,
                process_file=lambda raw_path, **_: seen.append(raw_path)
                or {
                    "raw_path": str(raw_path),
                    "filename": raw_path.name,
                    "model": "test-model",
                    "rating": {"rating": 3, "keep": True, "reason": "ok"},
                    "error": None,
                },
                emit=lambda _message: None,
            )

            self.assertEqual(seen, [second.resolve()])
            self.assertEqual(summary, {"processed": 1, "skipped": 1, "errors": 0})
