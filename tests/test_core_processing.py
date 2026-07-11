from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photo_cull_ai.core import (
    PreviewExtractionError,
    extract_embedded_jpeg,
    find_raw_files,
    process_raw_file,
    resize_to_temp_jpeg,
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

    def test_find_raw_files_deduplicates_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "photo.CR3"
            raw_path.write_text("", encoding="utf-8")
            (tmp_path / "alias.CR3").symlink_to(raw_path)

            self.assertEqual(find_raw_files(tmp_path), [raw_path.resolve()])

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

    def test_extract_embedded_jpeg_falls_back_after_first_command_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "IMG_0001.CR3"
            raw_path.write_text("", encoding="utf-8")
            calls = []

            def fake_run(args, **kwargs):
                calls.append((args, kwargs))
                if "-PreviewImage" in args:
                    return subprocess.CompletedProcess(args, 1, b"", b"Tag unavailable")
                return subprocess.CompletedProcess(args, 0, b"jpeg-data", b"")

            preview_path = extract_embedded_jpeg(raw_path, run_command=fake_run)
            try:
                self.assertEqual(preview_path.read_bytes(), b"jpeg-data")
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][1]["timeout"], 60)
            finally:
                preview_path.unlink(missing_ok=True)

    def test_resize_removes_output_temp_file_when_pillow_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preview_path = tmp_path / "invalid.jpg"
            preview_path.write_bytes(b"not-an-image")
            real_mkstemp = tempfile.mkstemp

            def fake_mkstemp(*_args, **_kwargs):
                return real_mkstemp(dir=tmp_path, prefix="resized-", suffix=".jpg")

            with mock.patch("photo_cull_ai.core.tempfile.mkstemp", side_effect=fake_mkstemp):
                with self.assertRaises(Exception):
                    resize_to_temp_jpeg(preview_path, 100)

            self.assertEqual(list(tmp_path.glob("resized-*.jpg")), [])

    def test_cleanup_failure_is_recorded_without_escaping(self):
        class LockedTempPath:
            def unlink(self, missing_ok=False):
                raise PermissionError("locked")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "IMG_0001.CR3"
            raw_path.write_text("", encoding="utf-8")
            output_path = tmp_path / "ratings.jsonl"
            locked_path = LockedTempPath()

            record = process_raw_file(
                raw_path=raw_path,
                output_path=output_path,
                model="test-model",
                max_size=100,
                extract_preview=lambda _path: locked_path,
                resize_preview=lambda _path, _size: locked_path,
                grade_image=lambda _path, _model: (
                    '{"rating": 3, "keep": true, "reason": "ok"}'
                ),
            )

            self.assertIn("Failed to remove temporary file", record["error"])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), record)

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
