"""
src/tax_talk/ingestion/loader.py

Step 1 of the ingestion pipeline:
    Loads PDFs from data/raw/<key_name>/ and returns (text, metadata) pairs.
    Each source folder has a metadata.json from download_corpus.py.

Usage:
    from tax_talk.ingestion.loader import load_all_sources, SourceDocument
    docs = load_all_sources()
    for doc in docs:
        print(doc.source_key, len(doc.text))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pypdf
from bs4 import BeautifulSoup

from tax_talk.core.runtime import get_logger

log = get_logger(__name__)

DATA_RAW = Path(__file__).parent.parent.parent.parent / "data" / "raw"


@dataclass
class SourceDocument:
    """One PDF file worth of text + all its metadata."""

    source_key: str
    file_path: Path
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def applicable_period(self) -> str:
        return self.metadata.get("applicable_period", "unknown")

    @property
    def act_name(self) -> str:
        return self.metadata.get("act_name", self.source_key)


def load_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF. Preserves page breaks as double newlines."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text.strip())
        full_text = "\n\n".join(pages)
        log.info("Loaded %s — %d pages, %d chars", pdf_path.name, len(pages), len(full_text))
        return full_text
    except Exception as e:
        log.error("Failed to read %s: %s", pdf_path, e)
        return ""


def load_html(html_path: Path) -> str:
    """Extract visible text from an HTML file."""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join([line for line in lines if line])
        log.info("Loaded %s — %d chars", html_path.name, len(cleaned))
        return cleaned
    except Exception as e:
        log.error("Failed to read %s: %s", html_path, e)
        return ""


def load_metadata(source_dir: Path) -> dict:
    """Load metadata.json from a source directory."""
    meta_path = source_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def load_source(source_key: str) -> list[SourceDocument]:
    """
    Load all supported documents from data/raw/<source_key>/.
    One SourceDocument per file.
    Returns empty list if directory doesn't exist or has no supported files.
    """
    source_dir = DATA_RAW / source_key

    if not source_dir.exists():
        log.warning("Source directory not found: %s", source_dir)
        return []

    metadata = load_metadata(source_dir)

    # Skip if it's a MANUAL or SKIP directory (no PDFs yet)
    if (source_dir / "MANUAL.txt").exists() or (source_dir / "SKIP.txt").exists():
        log.info("Skipping %s — manual download required or dead link.", source_key)
        return []

    supported_files = sorted(
        [
            p
            for p in source_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pdf", ".html", ".htm"}
        ]
    )
    if not supported_files:
        log.warning("No PDF/HTML files in %s", source_dir)
        return []

    docs = []
    for source_path in supported_files:
        if source_path.suffix.lower() == ".pdf":
            text = load_pdf(source_path)
        else:
            text = load_html(source_path)
        if not text:
            continue
        docs.append(
            SourceDocument(
                source_key=source_key,
                file_path=source_path,
                text=text,
                metadata={**metadata, "filename": source_path.name},
            )
        )

    log.info("Loaded %d document(s) from %s", len(docs), source_key)
    return docs


def load_all_sources(sources: list[str] | None = None) -> list[SourceDocument]:
    """
    Load all sources from data/raw/.
    Pass a list of source_keys to load specific ones only.

    Args:
        sources: optional list of source_key names. Loads all if None.

    Returns:
        list of SourceDocument objects, one per PDF file.
    """
    if not DATA_RAW.exists():
        log.error("data/raw/ not found. Run: uv run python scripts/download_corpus.py")
        return []

    if sources is None:
        # Auto-discover all subdirectories
        sources = [d.name for d in DATA_RAW.iterdir() if d.is_dir()]

    all_docs: list[SourceDocument] = []
    for key in sorted(sources):
        all_docs.extend(load_source(key))

    log.info("Total documents loaded: %d", len(all_docs))
    return all_docs


if __name__ == "__main__":
    docs = load_all_sources()
    for doc in docs:
        print(
            f"  {doc.source_key} | {doc.act_name} | {len(doc.text):,} chars | period: {doc.applicable_period}"
        )
