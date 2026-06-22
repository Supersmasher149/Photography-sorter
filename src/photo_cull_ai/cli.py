from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .core import run_batch, validate_startup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="photo-cull-ai")
    parser.add_argument("--input", required=True, help="Folder containing RAW files")
    parser.add_argument("--output", required=True, help="Path to ratings.jsonl")
    parser.add_argument(
        "--model",
        default="qwen2.5vl:3b",
        help="Ollama model name to use for grading",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="Maximum width or height for resized previews",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders for supported RAW files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already present in the output JSONL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_startup(
            input_dir=Path(args.input),
            output_path=Path(args.output),
            model=args.model,
            max_size=args.max_size,
        )
        run_batch(
            input_dir=Path(args.input),
            output_path=Path(args.output),
            model=args.model,
            max_size=args.max_size,
            recursive=args.recursive,
            resume=args.resume,
        )
    except Exception as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")

    return 0


def emit_stdout(message: str) -> None:
    print(message, file=sys.stdout)
