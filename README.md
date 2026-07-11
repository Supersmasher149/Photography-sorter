# Photo Culling CLI

Photo Culling CLI is a non-destructive Python command-line tool for AI-assisted RAW photo culling with a local Ollama vision model. It scans a folder of supported RAW files, extracts each file's embedded JPEG preview with `exiftool`, resizes the preview, sends it to Ollama for grading, writes one JSON object per file to a JSONL output file, and cleans up temporary preview files after each item.

## Safety Guarantees

- Never modifies RAW files
- Never deletes RAW files
- Never moves RAW files
- Rejects RAW files, including symlinks to RAW files, as `--output` targets
- Validates that `--output` is an appendable file before processing begins
- Only writes the validated output JSONL file and temporary preview JPEG files
- Removes temporary preview files even when processing fails

## Requirements

- Python 3.10 or newer
- `exiftool` available on `PATH`
- Ollama installed and running locally
- A local Ollama vision model such as `qwen2.5vl:3b`

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Or install the package in editable mode and get the console script:

```bash
python -m pip install -e .[dev]
```

## Install `exiftool`

### Windows

1. Download ExifTool for Windows from [exiftool.org](https://exiftool.org/).
2. Extract it and add the executable directory to your `PATH`.
3. Open a new terminal and verify:

```bash
exiftool -ver
```

### macOS

With Homebrew:

```bash
brew install exiftool
```

Verify:

```bash
exiftool -ver
```

## Install Ollama and Pull a Model

Install Ollama from [ollama.com](https://ollama.com/), start the local server, then pull a supported vision model:

```bash
ollama pull qwen2.5vl:3b
```

## Usage

Basic run:

```bash
python main.py --input "/path/to/raws" --output "/path/to/ratings.jsonl"
```

Installed console script:

```bash
photo-cull-ai --input "/path/to/raws" --output "/path/to/ratings.jsonl"
```

Recursive scan:

```bash
python main.py --input "/path/to/raws" --output "/path/to/ratings.jsonl" --recursive
```

Resume from an existing JSONL file:

```bash
python main.py --input "/path/to/raws" --output "/path/to/ratings.jsonl" --resume
```

Use a different model or resize limit:

```bash
python main.py --input "/path/to/raws" --output "/path/to/ratings.jsonl" --model llava:13b --max-size 1600
```

## Supported RAW Extensions

The CLI scans these case-insensitive extensions:

- `.cr2`
- `.cr3`
- `.nef`
- `.arw`
- `.dng`
- `.orf`
- `.rw2`
- `.raf`

## Output Format

Successful records look like this:

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

Error records keep the failure message, and invalid model JSON also preserves the raw model response for later review:

```json
{
  "raw_path": "/absolute/path/to/file.CR3",
  "filename": "file.CR3",
  "model": "qwen2.5vl:3b",
  "rating": null,
  "error": "Invalid JSON response: Expecting value",
  "raw_response": "not-json"
}
```

## Notes

- `--resume` treats any existing `raw_path` in the output file as completed, even if the earlier record contains an error.
- Invalid JSONL lines in resume mode stop startup with a clear error so you can repair the file intentionally.
- Model output must be strict JSON with `rating`, `keep`, and `reason`.
- The tool tries ExifTool's `PreviewImage` first, then falls back to `JpgFromRaw` when needed. Each extraction has a 60-second timeout.
- Symlink aliases to the same RAW file are processed only once.

## Run Tests

Install development dependencies, then run the full deterministic test suite:

```bash
python -m pip install -e .[dev]
python -m pytest
```

The tests mock ExifTool and Ollama, so they do not require a running Ollama server, a local model, or installed ExifTool.
