# Photo Culling CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cross-platform Python CLI that grades RAW photo previews with a local Ollama vision model without ever modifying the original RAW files.

**Architecture:** A thin top-level `main.py` will call into a small package under `src/photo_cull_ai`. Core file discovery, JSONL resume handling, preview extraction, resizing, model calls, and record writing will live in focused functions so the CLI and tests stay easy to reason about. Tests will mock external dependencies (`exiftool` and Ollama) while exercising real parsing and file-output behavior.

**Tech Stack:** Python 3.10+, argparse, pathlib, tempfile, subprocess, json, Pillow, ollama, pytest

---

### Task 1: Scaffold Package and Test Layout

**Files:**
- Create: `C:\Users\wally\Documents\photogrphay\main.py`
- Create: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\__init__.py`
- Create: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\cli.py`
- Create: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Create: `C:\Users\wally\Documents\photogrphay\tests\conftest.py`
- Create: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`
- Create: `C:\Users\wally\Documents\photogrphay\tests\test_cli.py`

- [ ] **Step 1: Write the failing import smoke test**

```python
from photo_cull_ai.cli import main


def test_main_is_importable() -> None:
    assert callable(main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py::test_main_is_importable -v`
Expected: FAIL with `ModuleNotFoundError` for `photo_cull_ai`

- [ ] **Step 3: Write minimal package scaffolding**

```python
# main.py
from photo_cull_ai.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# src/photo_cull_ai/__init__.py
__all__ = ["__version__"]

__version__ = "0.1.0"
```

```python
# src/photo_cull_ai/cli.py
def main() -> int:
    return 0
```

```python
# src/photo_cull_ai/core.py
# Core functions will be added in later tasks.
```

```python
# tests/conftest.py
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py::test_main_is_importable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py src/photo_cull_ai/__init__.py src/photo_cull_ai/cli.py src/photo_cull_ai/core.py tests/conftest.py tests/test_cli.py
git commit -m "chore: scaffold photo culling package"
```

### Task 2: Add JSONL Append and Resume Helpers

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`

- [ ] **Step 1: Write the failing tests for append and resume**

```python
import json
from pathlib import Path

import pytest

from photo_cull_ai.core import append_jsonl, load_completed_paths


def test_append_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    output_path = tmp_path / "ratings.jsonl"
    append_jsonl(output_path, {"raw_path": "one.CR3", "error": None})
    append_jsonl(output_path, {"raw_path": "two.CR3", "error": "bad"})

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["raw_path"] == "one.CR3"
    assert json.loads(lines[1])["raw_path"] == "two.CR3"


def test_load_completed_paths_returns_all_recorded_raw_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "ratings.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"raw_path": "/photos/A.CR3"}),
                json.dumps({"raw_path": "/photos/B.NEF", "error": "No preview"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_completed_paths(output_path) == {"/photos/A.CR3", "/photos/B.NEF"}


def test_load_completed_paths_raises_on_invalid_json_line(tmp_path: Path) -> None:
    output_path = tmp_path / "ratings.jsonl"
    output_path.write_text('{"raw_path": "/photos/A.CR3"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_completed_paths(output_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -k "append_jsonl or load_completed_paths" -v`
Expected: FAIL with `ImportError` or missing function errors

- [ ] **Step 3: Write minimal implementation for append and resume**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {output_path} at line {line_number}") from exc
            raw_path = record.get("raw_path")
            if isinstance(raw_path, str) and raw_path:
                completed.add(raw_path)
    return completed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -k "append_jsonl or load_completed_paths" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/photo_cull_ai/core.py tests/test_core.py
git commit -m "feat: add jsonl append and resume helpers"
```

### Task 3: Add Strict Model JSON Parsing

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`

- [ ] **Step 1: Write the failing tests for strict model parsing**

```python
import pytest

from photo_cull_ai.core import parse_model_json


def test_parse_model_json_accepts_valid_rating_payload() -> None:
    payload = parse_model_json('{"rating": 4, "keep": true, "reason": "Strong frame"}')

    assert payload == {"rating": 4, "keep": True, "reason": "Strong frame"}


@pytest.mark.parametrize(
    "raw_response",
    [
        "not json",
        '{"rating": 9, "keep": true, "reason": "bad"}',
        '{"rating": 4, "keep": "yes", "reason": "bad"}',
        '{"rating": 4, "keep": true}',
    ],
)
def test_parse_model_json_rejects_invalid_payloads(raw_response: str) -> None:
    with pytest.raises(ValueError):
        parse_model_json(raw_response)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -k "parse_model_json" -v`
Expected: FAIL with missing `parse_model_json`

- [ ] **Step 3: Write minimal implementation for response validation**

```python
def parse_model_json(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("Model returned invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")

    rating = payload.get("rating")
    keep = payload.get("keep")
    reason = payload.get("reason")

    if not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError("Model response rating must be an integer from 1 to 5")
    if not isinstance(keep, bool):
        raise ValueError("Model response keep must be a boolean")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Model response reason must be a non-empty string")

    return {"rating": rating, "keep": keep, "reason": reason.strip()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -k "parse_model_json" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/photo_cull_ai/core.py tests/test_core.py
git commit -m "feat: validate model json responses"
```

### Task 4: Add RAW Discovery

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`

- [ ] **Step 1: Write the failing tests for file discovery**

```python
from pathlib import Path

from photo_cull_ai.core import find_raw_files


def test_find_raw_files_returns_sorted_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "b.CR3").write_bytes(b"raw")
    (tmp_path / "a.nef").write_bytes(b"raw")
    (tmp_path / "skip.jpg").write_bytes(b"jpg")

    paths = find_raw_files(tmp_path, recursive=False)

    assert [path.name for path in paths] == ["a.nef", "b.CR3"]


def test_find_raw_files_honors_recursive_flag(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.arw").write_bytes(b"raw")

    assert find_raw_files(tmp_path, recursive=False) == []
    assert [path.name for path in find_raw_files(tmp_path, recursive=True)] == ["inside.arw"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -k "find_raw_files" -v`
Expected: FAIL with missing `find_raw_files`

- [ ] **Step 3: Write minimal implementation for deterministic RAW discovery**

```python
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf"}


def find_raw_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    candidates = (path for path in input_dir.glob(pattern) if path.is_file())
    raw_files = [path.resolve() for path in candidates if path.suffix.lower() in RAW_EXTENSIONS]
    return sorted(raw_files, key=lambda path: str(path).lower())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -k "find_raw_files" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/photo_cull_ai/core.py tests/test_core.py
git commit -m "feat: discover supported raw files"
```

### Task 5: Add Preview Extraction and Resize Helpers

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`

- [ ] **Step 1: Write the failing tests for extraction fallback and resize**

```python
import subprocess
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from photo_cull_ai.core import extract_embedded_jpeg, resize_to_temp_jpeg


def test_extract_embedded_jpeg_falls_back_to_jpg_from_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_path = tmp_path / "frame.CR3"
    raw_path.write_bytes(b"raw")

    preview_bytes = b"\xff\xd8\xff\xd9"
    responses = [
        subprocess.CompletedProcess(args=["exiftool"], returncode=0, stdout=b"", stderr=b""),
        subprocess.CompletedProcess(args=["exiftool"], returncode=0, stdout=preview_bytes, stderr=b""),
    ]

    def fake_run(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("photo_cull_ai.core.subprocess.run", fake_run)

    preview_path = extract_embedded_jpeg(raw_path)
    try:
        assert preview_path.read_bytes() == preview_bytes
    finally:
        preview_path.unlink(missing_ok=True)


def test_resize_to_temp_jpeg_limits_max_dimension(tmp_path: Path) -> None:
    source = tmp_path / "preview.jpg"
    Image.new("RGB", (4000, 2000), "white").save(source, format="JPEG")

    resized_path = resize_to_temp_jpeg(source, max_size=2048)
    try:
        with Image.open(resized_path) as image:
            assert max(image.size) == 2048
    finally:
        resized_path.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py -k "extract_embedded_jpeg or resize_to_temp_jpeg" -v`
Expected: FAIL with missing helper functions

- [ ] **Step 3: Write minimal implementation for extraction fallback and resize**

```python
import shutil
import subprocess
import tempfile

from PIL import Image, ImageOps


def extract_embedded_jpeg(raw_path: Path) -> Path:
    for tag in ("PreviewImage", "JpgFromRaw"):
        result = subprocess.run(
            ["exiftool", f"-{tag}", "-b", str(raw_path)],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "exiftool failed")
        if result.stdout:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as handle:
                handle.write(result.stdout)
                return Path(handle.name)
    raise RuntimeError("No embedded JPEG preview found.")


def resize_to_temp_jpeg(preview_path: Path, max_size: int) -> Path:
    with Image.open(preview_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")
        image.thumbnail((max_size, max_size))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as handle:
            image.save(handle, format="JPEG", quality=88)
            return Path(handle.name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py -k "extract_embedded_jpeg or resize_to_temp_jpeg" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/photo_cull_ai/core.py tests/test_core.py
git commit -m "feat: extract and resize raw previews"
```

### Task 6: Add Ollama Grading and Startup Validation

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\cli.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_cli.py`

- [ ] **Step 1: Write the failing tests for grading and prerequisite validation**

```python
from pathlib import Path

import pytest

from photo_cull_ai.cli import build_parser, validate_startup
from photo_cull_ai.core import grade_with_ollama


def test_grade_with_ollama_returns_message_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "preview.jpg"
    image_path.write_bytes(b"jpeg")

    class FakeClient:
        def chat(self, **kwargs):
            return {"message": {"content": '{"rating": 3, "keep": true, "reason": "Usable"}'}}

    monkeypatch.setattr("photo_cull_ai.core.Client", lambda host=None: FakeClient())

    raw_response = grade_with_ollama(image_path, model="qwen2.5vl:3b")

    assert raw_response == '{"rating": 3, "keep": true, "reason": "Usable"}'


def test_validate_startup_rejects_missing_input_directory(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["--input", str(tmp_path / "missing"), "--output", str(tmp_path / "ratings.jsonl")])

    with pytest.raises(SystemExit):
        validate_startup(args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core.py tests/test_cli.py -k "grade_with_ollama or validate_startup" -v`
Expected: FAIL with missing functions or imports

- [ ] **Step 3: Write minimal implementation for grading and startup checks**

```python
# src/photo_cull_ai/core.py
from ollama import Client

PROMPT = (
    "Strictly rate this photo.\n"
    "1=Reject, 2=Archive, 3=Keep, 4=Edit, 5=Portfolio.\n"
    "Return JSON only:\n"
    '{"rating":1-5,"keep":true,"reason":"short explanation"}'
)


def grade_with_ollama(image_path: Path, model: str) -> str:
    client = Client()
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0, "num_ctx": 1024},
    )
    content = response.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response did not contain message content")
    return content.strip()
```

```python
# src/photo_cull_ai/cli.py
import argparse
import shutil
from pathlib import Path

from ollama import Client


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cull RAW photos with a local Ollama vision model.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="qwen2.5vl:3b")
    parser.add_argument("--max-size", default=2048, type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_startup(args: argparse.Namespace) -> None:
    if not args.input.exists() or not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    if args.max_size <= 0:
        raise SystemExit("--max-size must be a positive integer")
    if shutil.which("exiftool") is None:
        raise SystemExit("exiftool is required on PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = Client()
    client.list()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core.py tests/test_cli.py -k "grade_with_ollama or validate_startup" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/photo_cull_ai/core.py src/photo_cull_ai/cli.py tests/test_core.py tests/test_cli.py
git commit -m "feat: add ollama grading and startup validation"
```

### Task 7: Build End-to-End CLI Orchestration

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\cli.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_cli.py`

- [ ] **Step 1: Write the failing CLI orchestration tests**

```python
import json
from pathlib import Path

from photo_cull_ai.cli import main


def test_main_processes_one_file_and_appends_rating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    raw_file = tmp_path / "frame.CR3"
    raw_file.write_bytes(b"raw")
    output_path = tmp_path / "ratings.jsonl"
    preview_path = tmp_path / "preview.jpg"
    preview_path.write_bytes(b"preview")
    resized_path = tmp_path / "resized.jpg"
    resized_path.write_bytes(b"resized")

    monkeypatch.setattr("photo_cull_ai.cli.validate_startup", lambda args: None)
    monkeypatch.setattr("photo_cull_ai.cli.find_raw_files", lambda input_dir, recursive: [raw_file])
    monkeypatch.setattr("photo_cull_ai.cli.extract_embedded_jpeg", lambda raw_path: preview_path)
    monkeypatch.setattr("photo_cull_ai.cli.resize_to_temp_jpeg", lambda preview, max_size: resized_path)
    monkeypatch.setattr("photo_cull_ai.cli.grade_with_ollama", lambda image_path, model: '{"rating": 5, "keep": true, "reason": "Best"}')

    exit_code = main(["--input", str(tmp_path), "--output", str(output_path)])

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    assert records[0]["rating"]["rating"] == 5


def test_main_preserves_raw_response_on_invalid_model_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_file = tmp_path / "frame.CR3"
    raw_file.write_bytes(b"raw")
    output_path = tmp_path / "ratings.jsonl"
    preview_path = tmp_path / "preview.jpg"
    preview_path.write_bytes(b"preview")
    resized_path = tmp_path / "resized.jpg"
    resized_path.write_bytes(b"resized")

    monkeypatch.setattr("photo_cull_ai.cli.validate_startup", lambda args: None)
    monkeypatch.setattr("photo_cull_ai.cli.find_raw_files", lambda input_dir, recursive: [raw_file])
    monkeypatch.setattr("photo_cull_ai.cli.extract_embedded_jpeg", lambda raw_path: preview_path)
    monkeypatch.setattr("photo_cull_ai.cli.resize_to_temp_jpeg", lambda preview, max_size: resized_path)
    monkeypatch.setattr("photo_cull_ai.cli.grade_with_ollama", lambda image_path, model: "not-json")

    main(["--input", str(tmp_path), "--output", str(output_path)])

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["rating"] is None
    assert record["raw_response"] == "not-json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -k "processes_one_file or preserves_raw_response" -v`
Expected: FAIL because `main()` does not orchestrate the workflow yet

- [ ] **Step 3: Write minimal orchestration implementation**

```python
import argparse
from pathlib import Path

from photo_cull_ai.core import (
    append_jsonl,
    extract_embedded_jpeg,
    find_raw_files,
    grade_with_ollama,
    load_completed_paths,
    parse_model_json,
    resize_to_temp_jpeg,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_startup(args)

    raw_files = find_raw_files(args.input.resolve(), recursive=args.recursive)
    completed = load_completed_paths(args.output.resolve()) if args.resume else set()
    pending = [path for path in raw_files if str(path) not in completed]

    processed = 0
    skipped = len(raw_files) - len(pending)
    errors = 0

    for index, raw_path in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] Processing {raw_path.name}")
        preview_path: Path | None = None
        resized_path: Path | None = None
        raw_response: str | None = None

        try:
            preview_path = extract_embedded_jpeg(raw_path)
            resized_path = resize_to_temp_jpeg(preview_path, max_size=args.max_size)
            raw_response = grade_with_ollama(resized_path, model=args.model)
            rating = parse_model_json(raw_response)
            record = {
                "raw_path": str(raw_path),
                "filename": raw_path.name,
                "model": args.model,
                "rating": rating,
                "error": None,
            }
            print(f"  rating={rating['rating']} keep={rating['keep']} reason={rating['reason']}")
        except Exception as exc:
            errors += 1
            record = {
                "raw_path": str(raw_path),
                "filename": raw_path.name,
                "model": args.model,
                "rating": None,
                "error": str(exc),
            }
            if raw_response is not None:
                record["raw_response"] = raw_response
            print(f"  error={exc}")
        finally:
            if resized_path is not None:
                resized_path.unlink(missing_ok=True)
            if preview_path is not None:
                preview_path.unlink(missing_ok=True)

        append_jsonl(args.output.resolve(), record)
        processed += 1

    print(f"Summary: processed={processed} skipped={skipped} errors={errors}")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -k "processes_one_file or preserves_raw_response" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/photo_cull_ai/cli.py tests/test_cli.py
git commit -m "feat: orchestrate raw photo grading cli"
```

### Task 8: Add Dependency Metadata and User Documentation

**Files:**
- Create: `C:\Users\wally\Documents\photogrphay\requirements.txt`
- Create: `C:\Users\wally\Documents\photogrphay\README.md`

- [ ] **Step 1: Write the failing documentation existence checks**

```python
from pathlib import Path


def test_readme_exists() -> None:
    assert Path("README.md").is_file()


def test_requirements_exists() -> None:
    assert Path("requirements.txt").is_file()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -k "readme_exists or requirements_exists" -v`
Expected: FAIL because the files do not exist yet

- [ ] **Step 3: Write minimal dependency and setup documentation**

```text
# requirements.txt
ollama
Pillow
pytest
```

```markdown
# Photo Cull AI CLI

CLI tool for non-destructive AI photo culling using local Ollama vision models. It reads supported RAW files, extracts embedded JPEG previews with `exiftool`, resizes those previews, sends them to Ollama for grading, writes one JSON object per line to `ratings.jsonl`, and deletes all temporary JPEG files after use.

## Safety

- Never modifies RAW files
- Never moves or deletes RAW files
- Only writes the output JSONL file and temporary JPEG previews

## Requirements

- Python 3.10+
- `exiftool` available on `PATH`
- Ollama installed and running locally
- A pulled vision model such as `qwen2.5vl:3b`

## Setup

### Windows

1. Install Python 3.10 or newer.
2. Install Ollama from [https://ollama.com](https://ollama.com).
3. Install `exiftool` and make sure the executable is available on `PATH`.
4. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

### macOS

1. Install Python 3.10 or newer.
2. Install Ollama from [https://ollama.com](https://ollama.com).
3. Install `exiftool`, for example with Homebrew:

```bash
brew install exiftool
```

4. Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Pull a Model

```bash
ollama pull qwen2.5vl:3b
```

## Example Commands

```bash
python main.py --input /path/to/raws --output ratings.jsonl
python main.py --input /path/to/raws --output ratings.jsonl --recursive
python main.py --input /path/to/raws --output ratings.jsonl --resume
python main.py --input /path/to/raws --output ratings.jsonl --model qwen2.5vl:3b --max-size 1600
```

## Output Notes

If the model returns invalid JSON, the tool preserves `raw_response` in the JSONL record and continues processing the remaining files.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -k "readme_exists or requirements_exists" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt tests/test_cli.py
git commit -m "docs: add setup and usage instructions"
```

### Task 9: Run Full Verification

**Files:**
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\cli.py`
- Modify: `C:\Users\wally\Documents\photogrphay\src\photo_cull_ai\core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_core.py`
- Modify: `C:\Users\wally\Documents\photogrphay\tests\test_cli.py`
- Modify: `C:\Users\wally\Documents\photogrphay\README.md`
- Modify: `C:\Users\wally\Documents\photogrphay\requirements.txt`

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -v`
Expected: PASS with all tests green

- [ ] **Step 2: Run the CLI help output**

Run: `python main.py --help`
Expected: exit code 0 and usage text showing `--input`, `--output`, `--model`, `--max-size`, `--recursive`, and `--resume`

- [ ] **Step 3: Review the implementation against the spec**

Checklist:
- Supported RAW extensions match the spec
- Extraction tries `PreviewImage` before `JpgFromRaw`
- Resize uses Pillow auto-orientation and max-size bounding
- Ollama request uses one image, fresh request, and `temperature=0`
- Invalid model JSON preserves `raw_response`
- Output JSONL remains resumable
- Temporary files are deleted in `finally` blocks
- README includes Windows and macOS setup plus example commands

- [ ] **Step 4: Commit final cleanup if needed**

```bash
git add src/photo_cull_ai/cli.py src/photo_cull_ai/core.py tests/test_core.py tests/test_cli.py README.md requirements.txt
git commit -m "test: verify photo culling cli behavior"
```
