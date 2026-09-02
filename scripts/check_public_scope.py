#!/usr/bin/env python3
"""Reject development-only material from the public artifact tree."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".gitignore", ".json", ".md", ".py", ".toml", ".txt", ".yml"}
FORBIDDEN_TEXT = (
    "/" + "home/",
    "gmmd-" + "v" + "10",
    "v" + "10.1",
    "GMMD-" + "K1",
    "rank" + "split",
    "gpu-" + "edward",
    "dgx-" + "spark",
    "retained_" + "information",
)
FORBIDDEN_PATH_PARTS = {
    "history",
    "reviews",
    "conversations",
    "per_image",
    "submission",
}


def main() -> int:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or "__pycache__" in relative.parts or ".pytest_cache" in relative.parts:
            continue
        if any(part.lower() in FORBIDDEN_PATH_PARTS for part in relative.parts):
            failures.append(f"forbidden path: {relative.as_posix()}")
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"SHA256SUMS", "LICENSE"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for fragment in FORBIDDEN_TEXT:
            if fragment.lower() in text.lower():
                failures.append(f"forbidden text {fragment!r}: {relative.as_posix()}")
    for pattern in ("*.tex", "assembled_paper*", "main.pdf"):
        for path in ROOT.rglob(pattern):
            if ".git" not in path.parts:
                failures.append(f"manuscript or submission artifact: {path.relative_to(ROOT)}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"PUBLIC SCOPE CHECK: FAIL ({len(failures)})")
        return 1
    print("PUBLIC SCOPE CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
