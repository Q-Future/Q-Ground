#!/usr/bin/env python3
"""Extract all .tgz/.tar archives from a directory."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import os
import threading
import tarfile
from pathlib import Path, PurePosixPath


SUPPORTED_SUFFIXES = (".tar", ".tgz", ".tar.gz")
LOG_LOCK = threading.Lock()


def log_line(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def find_archives(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in root.rglob("*") if p.is_file() and is_supported_archive(p))
    return sorted(p for p in root.iterdir() if p.is_file() and is_supported_archive(p))


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination_abs = destination.resolve()
    destination_name = destination.name.lower()

    for member in tar.getmembers():
        member_parts = PurePosixPath(member.name).parts
        while len(member_parts) > 1 and member_parts[0].lower() == destination_name:
            member_parts = member_parts[1:]

        if not member_parts:
            continue

        normalized_name = str(PurePosixPath(*member_parts))
        member_path = (destination / normalized_name).resolve()
        if os.path.commonpath([destination_abs, member_path]) != str(destination_abs):
            raise RuntimeError(f"Blocked path traversal attempt in archive member: {member.name}")

        normalized_member = copy.copy(member)
        normalized_member.name = normalized_name
        tar.extract(normalized_member, path=destination)


def extract_archive(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:*") as tar:
        safe_extract(tar, destination)


class ProgressBar:
    def __init__(self, total: int, width: int = 32) -> None:
        self.total = max(total, 1)
        self.width = width
        self.current = 0
        self._lock = threading.Lock()

    def update(self, step: int = 1) -> None:
        with self._lock:
            self.current = min(self.total, self.current + step)
            ratio = self.current / self.total
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            print(
                f"\rProgress [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)",
                end="",
                flush=True,
            )

    def finish(self) -> None:
        with self._lock:
            print()


def compute_destination(archive: Path, dest_mode: str) -> Path:
    if dest_mode == "same-dir":
        return archive.parent

    stem = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    dest = archive.parent / stem
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def extract_one(archive: Path, dest_mode: str) -> tuple[Path, bool, str]:
    dest = compute_destination(archive, dest_mode)
    log_line(f"Extracting: {archive} -> {dest}")
    try:
        extract_archive(archive, dest)
        return archive, True, f"Extracted: {archive} -> {dest}"
    except Exception as exc:
        return archive, False, f"Failed: {archive} ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract all .tgz/.tar archives under a directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./dataset"),
        help="Directory to scan for archives (default: ./dataset).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top-level directory.",
    )
    parser.add_argument(
        "--dest-mode",
        choices=("same-dir", "subdir"),
        default="same-dir",
        help="Extraction destination: same-dir extracts beside each archive, subdir extracts into a folder named after archive stem.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of worker threads (default: half of CPU cores).",
    )

    args = parser.parse_args()
    root = args.root

    if not root.exists() or not root.is_dir():
        print(f"Error: {root} is not a valid directory")
        return 1

    archives = find_archives(root, recursive=not args.non_recursive)
    if not archives:
        print(f"No archives found in {root}")
        return 0

    workers = max(1, args.workers)
    print(f"Found {len(archives)} archives")
    print(f"Using {workers} worker threads")
    extracted = 0
    failed = 0
    messages: list[str] = []
    progress = ProgressBar(total=len(archives))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_one, archive, args.dest_mode) for archive in archives]

        for future in concurrent.futures.as_completed(futures):
            archive, ok, msg = future.result()
            if ok:
                extracted += 1
            else:
                failed += 1
            messages.append(msg)
            progress.update()

    progress.finish()

    for msg in messages:
        print(msg)

    print(f"Done. Extracted: {extracted}, Failed: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
