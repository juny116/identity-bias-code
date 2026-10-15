from identity_bias.runtime.paths import resolve_path

"""Streaming JSONL records with explicit handling of malformed lines."""

import json
from pathlib import Path


def read_records(path, *, skip_invalid=False):
    with resolve_path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                if not skip_invalid:
                    raise ValueError(f"{path}:{number}: invalid JSON") from error


def read_directory(directory, *, skip_invalid=True):
    for path in sorted(Path(directory).glob("*.jsonl")):
        yield from read_records(path, skip_invalid=skip_invalid)
