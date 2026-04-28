import argparse
import json
import sys
from pathlib import Path

TEXT_MIME_TYPES = {
    "text/plain",
    "text/html",
    "text/markdown",
    "text/latex",
    "application/javascript",
}

IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "application/pdf",
}

TRIM_MARKER = "... [output trimmed by contrib/trace_analysis/trim_output.sh]"
IMAGE_MARKER = (
    "... [embedded image/PDF output removed by "
    "contrib/trace_analysis/trim_output.sh]"
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="trim_output.sh",
        description="Trim large outputs in Jupyter notebooks without clearing every result."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Notebook files or directories to process. Directories are searched recursively.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=80,
        help="Maximum number of lines to keep for text-like outputs. Default: 80.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=20000,
        help="Maximum number of characters to keep for text-like outputs. Default: 20000.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove all cell outputs and execution counts instead of trimming.",
    )
    parser.add_argument(
        "--drop-images",
        action="store_true",
        default=True,
        help="Replace embedded image/PDF payloads with a short placeholder. This is the default.",
    )
    parser.add_argument(
        "--keep-images",
        action="store_false",
        dest="drop_images",
        help="Keep embedded image/PDF payloads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with status 1 if any notebook would change.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=1,
        help="JSON indentation used when writing notebooks. Default: 1.",
    )
    return parser.parse_args()


def iter_notebooks(paths):
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = sorted(path.rglob("*.ipynb"))
        else:
            candidates = [path]

        for candidate in candidates:
            if candidate.suffix != ".ipynb":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def as_text(value):
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    if isinstance(value, str):
        return value
    return None


def restore_text(original, text):
    if isinstance(original, list):
        return text.splitlines(keepends=True)
    return text


def trim_text(value, max_lines, max_chars):
    text = as_text(value)
    if text is None:
        return value, False
    if TRIM_MARKER in text or IMAGE_MARKER in text:
        return value, False

    lines = text.splitlines(keepends=True)
    changed = False

    if max_lines >= 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        text = "".join(lines)
        changed = True

    if max_chars >= 0 and len(text) > max_chars:
        text = text[:max_chars]
        changed = True

    if not changed:
        return value, False

    text = text.rstrip("\n")
    text += f"\n{TRIM_MARKER}\n"
    return restore_text(value, text), True


def trim_mime_bundle(data, args):
    changed = False
    if not isinstance(data, dict):
        return changed

    for mime_type in sorted(TEXT_MIME_TYPES & data.keys()):
        trimmed, did_change = trim_text(data[mime_type], args.max_lines, args.max_chars)
        if did_change:
            data[mime_type] = trimmed
            changed = True

    if args.drop_images:
        dropped = sorted(IMAGE_MIME_TYPES & data.keys())
        for mime_type in dropped:
            del data[mime_type]
            changed = True
        if dropped and "text/plain" not in data:
            data["text/plain"] = f"{IMAGE_MARKER}\n"

    return changed


def trim_notebook(notebook, args):
    changed = False
    cells = notebook.get("cells", [])

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        if args.clear:
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True
            continue

        for output in cell.get("outputs", []):
            output_type = output.get("output_type")

            if output_type == "stream" and "text" in output:
                trimmed, did_change = trim_text(
                    output["text"], args.max_lines, args.max_chars
                )
                if did_change:
                    output["text"] = trimmed
                    changed = True

            elif output_type in {"display_data", "execute_result"}:
                if trim_mime_bundle(output.get("data"), args):
                    changed = True

            elif output_type == "error" and "traceback" in output:
                trimmed, did_change = trim_text(
                    output["traceback"], args.max_lines, args.max_chars
                )
                if did_change:
                    output["traceback"] = trimmed
                    changed = True

    return changed


def main():
    args = parse_args()
    notebooks = list(iter_notebooks(args.paths))
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 2

    changed_paths = []
    failed = False

    for path in notebooks:
        try:
            original = path.read_text(encoding="utf-8")
            notebook = json.loads(original)
        except Exception as exc:
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            failed = True
            continue

        changed = trim_notebook(notebook, args)
        if not changed:
            print(f"ok      {path}")
            continue

        changed_paths.append(path)
        if args.dry_run or args.check:
            print(f"trim    {path}")
            continue

        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=args.indent) + "\n",
            encoding="utf-8",
        )
        print(f"trimmed {path}")

    if failed:
        return 2
    if args.check and changed_paths:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())