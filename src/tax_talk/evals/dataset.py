from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenQASample(BaseModel):
    """Single hand-labeled eval sample."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    expected_citations: list[str] = Field(default_factory=list)
    difficulty: str = "unknown"
    category: str = "unknown"


def load_golden_dataset(path: Path) -> list[GoldenQASample]:
    """Load and validate JSONL golden QA samples.

    Args:
        path: Absolute or relative JSONL dataset path.

    Returns:
        Parsed and validated dataset entries.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[GoldenQASample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            samples.append(GoldenQASample.model_validate(payload))

    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    return samples
