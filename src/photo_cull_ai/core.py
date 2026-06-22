from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


SUPPORTED_RAW_EXTENSIONS = {
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".orf",
    ".rw2",
    ".raf",
}

OLLAMA_PROMPT = (
    "Strictly rate this photo. 1=Reject, 2=Archive, 3=Keep, 4=Edit, "
    '5=Portfolio. Return JSON only: {"rating":1-5,"keep":true,'
    '"reason":"short explanation"}'
)


class PreviewExtractionError(RuntimeError):
    """Raised when a RAW file has no usable embedded JPEG preview."""


def find_raw_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    path = Path(input_dir)
    iterator = path.rglob("*") if recursive else path.iterdir()
    raw_files = [
        candidate.resolve()
        for candidate in iterator
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_RAW_EXTENSIONS
    ]
    return sorted(raw_files, key=lambda item: str(item).lower())


def load_completed_paths(output_path: Path) -> set[str]:
    path = Path(output_path)
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} on line {line_number}: {exc.msg}"
                ) from exc
            raw_path = record.get("raw_path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"Invalid JSONL in {path} on line {line_number}: missing raw_path"
                )
            completed.add(raw_path)
    return completed


def append_jsonl(output_path: Path, record: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_model_json(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")

    rating = payload.get("rating")
    if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
        raise ValueError("Model response rating must be an integer from 1 to 5")

    keep = payload.get("keep")
    if not isinstance(keep, bool):
        raise ValueError("Model response keep must be a boolean")

    reason = payload.get("reason")
    if not isinstance(reason, str):
        raise ValueError("Model response reason must be a string")

    return {"rating": rating, "keep": keep, "reason": reason}


def extract_embedded_jpeg(
    raw_path: Path,
    run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> Path:
    fd, temp_name = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        for tag in ("PreviewImage", "JpgFromRaw"):
            result = run_command(
                ["exiftool", f"-{tag}", "-b", str(raw_path)],
                check=False,
                capture_output=True,
            )
            if result.returncode != 0:
                stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    raise PreviewExtractionError(
                        f"exiftool failed for {raw_path.name}: {stderr_text}"
                    )
                continue
            data = result.stdout
            if data:
                temp_path.write_bytes(data)
                return temp_path
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    if temp_path.exists():
        temp_path.unlink()
    raise PreviewExtractionError("No embedded JPEG preview found.")


def resize_to_temp_jpeg(preview_path: Path, max_size: int) -> Path:
    from PIL import Image, ImageOps

    output_path = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
    with Image.open(preview_path) as image:
        oriented = ImageOps.exif_transpose(image)
        if oriented.mode != "RGB":
            oriented = oriented.convert("RGB")
        oriented.thumbnail((max_size, max_size))
        oriented.save(output_path, format="JPEG", quality=88)
    return output_path


def grade_with_ollama(image_path: Path, model: str, client: Any | None = None) -> str:
    if client is None:
        from ollama import Client

        client = Client()

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": OLLAMA_PROMPT,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": 0, "num_ctx": 512},
    )
    return _extract_message_content(response)


def _extract_message_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    raise ValueError("Ollama response did not contain message content")


def process_raw_file(
    raw_path: Path,
    output_path: Path,
    model: str,
    max_size: int,
    extract_preview: Callable[[Path], Path] = extract_embedded_jpeg,
    resize_preview: Callable[[Path, int], Path] = resize_to_temp_jpeg,
    grade_image: Callable[[Path, str], str] = grade_with_ollama,
) -> dict[str, Any]:
    preview_path: Path | None = None
    resized_path: Path | None = None
    raw_response: str | None = None
    record: dict[str, Any]

    try:
        preview_path = extract_preview(raw_path)
        resized_path = resize_preview(preview_path, max_size)
        raw_response = grade_image(resized_path, model)
        rating = parse_model_json(raw_response)
        record = build_success_record(raw_path=raw_path, model=model, rating=rating)
    except Exception as exc:
        record = build_error_record(
            raw_path=raw_path,
            model=model,
            error=str(exc),
            raw_response=raw_response,
        )
    finally:
        for temp_path in (preview_path, resized_path):
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    append_jsonl(output_path, record)
    return record


def build_success_record(raw_path: Path, model: str, rating: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_path": str(raw_path.resolve()),
        "filename": raw_path.name,
        "model": model,
        "rating": rating,
        "error": None,
    }


def build_error_record(
    raw_path: Path,
    model: str,
    error: str,
    raw_response: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "raw_path": str(raw_path.resolve()),
        "filename": raw_path.name,
        "model": model,
        "rating": None,
        "error": error,
    }
    if raw_response is not None:
        record["raw_response"] = raw_response
    return record


def run_batch(
    input_dir: Path,
    output_path: Path,
    model: str,
    max_size: int,
    recursive: bool,
    resume: bool,
    process_file: Callable[..., dict[str, Any]] = process_raw_file,
    emit: Callable[[str], None] = print,
) -> dict[str, int]:
    all_files = find_raw_files(input_dir, recursive=recursive)
    completed_paths = load_completed_paths(output_path) if resume else set()
    files_to_process = [
        raw_path for raw_path in all_files if str(raw_path.resolve()) not in completed_paths
    ]

    processed = 0
    skipped = len(all_files) - len(files_to_process)
    errors = 0
    total = len(files_to_process)

    for index, raw_path in enumerate(files_to_process, start=1):
        emit(f"[{index}/{total}] Processing {raw_path.name}")
        record = process_file(
            raw_path=raw_path.resolve(),
            output_path=output_path,
            model=model,
            max_size=max_size,
        )
        processed += 1

        if record.get("error"):
            errors += 1
            emit(f"Error: {raw_path.name}: {record['error']}")
        else:
            rating = record["rating"]["rating"]
            emit(f"Rating: {raw_path.name}: {rating}")

    summary = {"processed": processed, "skipped": skipped, "errors": errors}
    emit(
        "Summary: "
        f"processed={summary['processed']} "
        f"skipped={summary['skipped']} "
        f"errors={summary['errors']}"
    )
    return summary


def validate_startup(
    input_dir: Path,
    output_path: Path,
    model: str,
    max_size: int,
    exiftool_lookup: Callable[[str], str | None] = shutil.which,
    ollama_ready: Callable[[str], None] | None = None,
) -> None:
    if max_size <= 0:
        raise ValueError("--max-size must be a positive integer")

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError("--input must be an existing directory")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if exiftool_lookup("exiftool") is None:
        raise RuntimeError("exiftool was not found on PATH")

    if ollama_ready is None:
        ollama_ready = ensure_ollama_ready

    ollama_ready(model)


def ensure_ollama_ready(model: str, client: Any | None = None) -> None:
    if client is None:
        from ollama import Client

        client = Client()

    try:
        listing = client.list()
    except Exception as exc:
        raise RuntimeError("Unable to contact the local Ollama server") from exc

    available_models = _extract_model_names(listing)
    if model not in available_models:
        raise RuntimeError(f"Ollama model '{model}' is not available locally")


def _extract_model_names(listing: Any) -> set[str]:
    if isinstance(listing, dict):
        models = listing.get("models", [])
    else:
        models = getattr(listing, "models", [])

    names: set[str] = set()
    for model in models:
        if isinstance(model, dict):
            for key in ("model", "name"):
                value = model.get(key)
                if isinstance(value, str):
                    names.add(value)
        else:
            for key in ("model", "name"):
                value = getattr(model, key, None)
                if isinstance(value, str):
                    names.add(value)
    return names
