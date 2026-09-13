"""Auditable JSONL/JSON outputs for alignment-baseline selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class SelectionArtifactWriter:
    def __init__(self, root: str, method: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.method = method
        self.records_path = self.root / f"{method}_selection.jsonl"
        self.summary_path = self.root / f"{method}_summary.json"

    def write_records(self, records: Iterable[dict[str, Any]]) -> None:
        with self.records_path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def prompt_id(prompt: str) -> str:
    """Stable non-reversible identifier; raw prompts stay out of artifacts."""
    import hashlib

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
