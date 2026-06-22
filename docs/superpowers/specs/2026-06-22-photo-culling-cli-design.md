# Photo Culling CLI Design

## Goal

Build a cross-platform Python CLI tool for non-destructive AI photo culling using Ollama. The tool scans a folder of supported RAW photo files, extracts an embedded JPEG preview from each RAW file, resizes the preview to a bounded size, sends the resized image to a local Ollama vision model for grading, appends the result to a JSONL file, and deletes all temporary preview files. The original RAW files are never modified, moved, deleted, or overwritten.

## Scope

This design covers a CLI-only first version for Windows and macOS. It includes:

- RAW file discovery
- Resume support from an existing JSONL file
- Embedded preview extraction through `exiftool`
- Preview auto-orientation and resizing through Pillow
- Per-image grading through the Ollama Python package
- JSONL output for successful and failed items
- Minimal automated tests for core behavior
- Setup and usage documentation

This design does not include:

- A GUI
- Parallel processing
- Database storage
- RAW decoding beyond embedded preview extraction
- Cloud APIs or remote inference

## User Interface

The tool is invoked from the command line and supports these arguments:

- `--input`: required path to the folder containing RAW files
- `--output`: required path to `ratings.jsonl`
- `--model`: optional Ollama model name, default `qwen2.5vl:3b`
- `--max-size`: optional integer maximum dimension for preview resizing, default `2048`
- `--recursive`: optional flag to scan subfolders
- `--resume`: optional flag to skip files already present in the output JSONL

Runtime output should include:

- Progress lines in the form `[12/300] Processing IMG_1234.CR3`
- A result line after each file showing either the rating or the recorded error
- A final summary showing `processed`, `skipped`, and `errors`

## Repository Layout

The project uses a small packaged layout while still providing the required top-level `main.py`.

- `main.py`: thin entrypoint that imports and runs package `main()`
- `src/photo_cull_ai/__init__.py`: package marker
- `src/photo_cull_ai/cli.py`: CLI parsing and top-level orchestration
- `src/photo_cull_ai/core.py`: core logic functions
- `tests/`: minimal automated tests
- `requirements.txt`: runtime and test dependencies
- `README.md`: setup, prerequisites, and example commands

## Core Flow

1. Parse CLI arguments.
2. Validate startup prerequisites.
3. Discover supported RAW files from the input directory.
4. If `--resume` is enabled, load completed RAW paths from the output JSONL and filter them out.
5. For each remaining RAW file:
   - Extract an embedded JPEG preview using `exiftool`
   - If extraction fails or no preview exists, append an error record and continue
   - Resize the preview into a temporary JPEG using Pillow
   - Send the resized preview to Ollama for grading with a fresh request
   - Parse the model response as strict JSON
   - Append a JSONL record with either structured rating output or error details
   - Delete all temporary preview files in `finally` blocks
6. Print the final summary.

Each file is handled independently so crashes or one-off failures do not invalidate already written results.

## Startup Validation

Startup should fail fast before processing any files if prerequisites are missing or unusable:

- `exiftool` must be discoverable on `PATH`
- The Ollama client must be able to contact the local Ollama server
- The requested model must be available for inference
- `--input` must exist and be a directory
- The parent directory for `--output` must exist or be creatable inside the user-selected path
- `--max-size` must be a positive integer

Fail-fast behavior is intentional for this first version because the selected runtime mode is strict prerequisite enforcement.

## Supported RAW Files

The file scan supports these case-insensitive extensions:

- `.cr2`
- `.cr3`
- `.nef`
- `.arw`
- `.dng`
- `.orf`
- `.rw2`
- `.raf`

`find_raw_files()` returns deterministically ordered absolute paths so progress output and resume behavior are stable across runs.

## Extraction Design

`extract_embedded_jpeg(raw_path)` uses `subprocess.run()` to invoke `exiftool` without ever writing back to the RAW file.

Extraction order:

1. Try `-PreviewImage -b`
2. If empty or unavailable, try `-JpgFromRaw -b`

Implementation details:

- Extraction writes binary preview data to a temporary file, not alongside the RAW
- If `exiftool` returns no data for both tags, the function raises a controlled exception that the caller records as a per-file error
- The function never invokes any metadata-writing flags
- RAW input paths are passed as literals to avoid accidental globbing or shell expansion

## Resize Design

`resize_to_temp_jpeg(preview_path, max_size)` uses Pillow to:

- Open the extracted preview
- Apply EXIF-based auto-orientation with `ImageOps.exif_transpose`
- Convert to RGB if needed for JPEG output
- Resize using `thumbnail()` so neither width nor height exceeds `max_size`
- Save to a new temporary `.jpg` file with quality `88`

The original extracted preview file and the resized JPEG path are both temporary artifacts. The caller owns cleanup and removes them in `finally` blocks.

## Ollama Request Design

`grade_with_ollama(image_path, model)` sends one image per request through the Ollama Python package.

Request rules:

- One image per call
- Fresh request each time, no reused chat history
- `temperature=0`
- Low-context settings to avoid context buildup
- Prompt text is fixed:

`Strictly rate this photo. 1=Reject, 2=Archive, 3=Keep, 4=Edit, 5=Portfolio. Return JSON only: {"rating":1-5,"keep":true,"reason":"short explanation"}`

The function returns the raw model response text for downstream parsing so invalid JSON can be preserved.

## Response Parsing

`parse_model_json(raw_response)` attempts strict JSON decoding and validates:

- top-level object
- integer `rating` between 1 and 5
- boolean `keep`
- string `reason`

If validation succeeds, the normalized rating object is returned.

If validation fails, the caller records:

- `rating: null`
- `error`: descriptive parse/validation error
- `raw_response`: the raw model text

This preserves evidence for later manual review without stopping the batch.

## Output Format

`append_jsonl(output_path, record)` appends exactly one JSON object per line using UTF-8 encoding.

Successful record shape:

```json
{
  "raw_path": "/absolute/path/to/file.CR3",
  "filename": "file.CR3",
  "model": "qwen2.5vl:3b",
  "rating": {
    "rating": 4,
    "keep": true,
    "reason": "Strong composition and usable expression."
  },
  "error": null
}
```

Error record shape:

```json
{
  "raw_path": "/absolute/path/to/file.CR3",
  "filename": "file.CR3",
  "model": "qwen2.5vl:3b",
  "rating": null,
  "error": "No embedded JPEG preview found.",
  "raw_response": "..."
}
```

`raw_response` is included only when useful, especially for invalid model JSON. One append happens immediately after each file finishes processing so the output remains resumable after crashes.

## Resume Behavior

`load_completed_paths(output_path)` reads existing JSONL records and returns a set of `raw_path` values already present.

Rules:

- Missing output file means no completed paths
- Invalid JSONL lines should raise a clear startup error when `--resume` is requested
- A path counts as completed if it already appears in the output, regardless of whether the prior record contains a rating or an error

This keeps resume behavior simple and crash-safe. Reprocessing can still be forced by running without `--resume` or deleting lines from the JSONL file intentionally.

## Error Handling

Per-file errors are expected and should not stop the full batch for cases such as:

- No embedded preview found
- Preview extraction failure for a single file
- Image decode or resize failure
- Model response parse failure

Process-wide errors should stop execution before or during startup for cases such as:

- Missing `exiftool`
- Unreachable Ollama server
- Requested model unavailable
- Invalid CLI arguments
- Unreadable input directory

Unexpected exceptions during per-file processing should be caught, recorded in JSONL, counted in the summary, and processing should continue to the next file.

## Safety Guarantees

The implementation must preserve these safety properties:

- Never alter RAW files
- Never delete RAW files
- Never move RAW files
- Never overwrite RAW files
- Only write to the output JSONL and temporary preview JPEG files
- Always clean up temporary preview files in `finally` blocks

The design avoids shell redirection and avoids any write-capable metadata tools against the RAW files.

## Testing Strategy

The first pass includes minimal automated tests focused on the riskiest logic:

- `load_completed_paths()` reads prior JSONL and supports resume behavior
- `parse_model_json()` accepts valid JSON and rejects invalid or incomplete JSON
- `append_jsonl()` writes one line per record in append mode
- Orchestration preserves `raw_response` when model output is invalid

Tests should use mocks for `subprocess.run()` and Ollama client calls where external dependencies would otherwise be required. Integration tests against real `exiftool` or Ollama are out of scope for the first pass.

## Documentation

`README.md` should include:

- What the tool does and its safety guarantees
- Python version requirement
- Dependency installation
- `exiftool` installation instructions for Windows and macOS
- Ollama installation and model pull example
- Example CLI commands for normal runs, recursive scans, and resume mode
- Notes on how invalid JSON model responses are recorded

## Implementation Notes

The required functions will exist with these responsibilities:

- `find_raw_files()`: discover supported RAW files
- `load_completed_paths()`: load previously recorded `raw_path` values
- `extract_embedded_jpeg()`: create a temp preview from `exiftool`
- `resize_to_temp_jpeg()`: auto-orient and bound image size into a temp JPEG
- `grade_with_ollama()`: send one image to the configured model
- `parse_model_json()`: decode and validate strict JSON output
- `append_jsonl()`: append one record atomically at the application level
- `main()`: CLI entrypoint and orchestration

## Success Criteria

The first version is successful if:

- It runs on Python 3.10+ on Windows and macOS
- It never modifies original RAW files
- It produces one JSONL record per processed RAW file
- It can resume safely from prior output
- It cleans up temporary preview files even on failures
- It gives clear progress and summary output
- It includes the requested files and minimal test coverage
