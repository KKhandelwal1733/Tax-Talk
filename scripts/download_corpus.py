"""Download corpus sources into data/raw with metadata and skip markers.

By default, downloads sources listed in DOWNLOAD_PRIORITY from scripts/corpus_scripts.py.
Each source is saved under data/raw/<source_key>/ with:
  - downloaded file (pdf/html/etc)
  - metadata.json (per-source result + metadata)

An aggregate manifest is written to data/raw/download_manifest.json.

Usage:
  uv run python scripts/download_corpus.py
  uv run python scripts/download_corpus.py --all
  uv run python scripts/download_corpus.py --source it_act_2025_icai --source it_act_1961_indiacode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from corpus_scripts import (
    CHUNK_METADATA_TEMPLATE,
    DOWNLOAD_PRIORITY,
    OFFICIAL_SOURCES,
    USER_PROVIDED_URLS,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
MANIFEST_PATH = RAW_ROOT / "download_manifest.json"
USER_AGENT = "tax-talk-downloader/0.1"
TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1024 * 128


@dataclass(frozen=True)
class SourceDef:
    key: str
    url: str
    source_group: str
    source_meta: dict[str, Any]
    ingestion_metadata: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_ext(content_type: str | None, url: str) -> str:
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip().lower())
        if guessed:
            return guessed

    path = unquote(urlparse(url).path)
    suffix = Path(path).suffix.lower()
    if suffix in {".pdf", ".html", ".htm", ".txt", ".json", ".xml"}:
        return suffix
    return ".bin"


def _filename_from_headers(resp: requests.Response) -> str | None:
    cd = resp.headers.get("content-disposition", "")
    if "filename=" not in cd.lower():
        return None

    # Handles filename="x.pdf" and filename*=UTF-8''x.pdf variants loosely.
    parts = cd.split(";")
    for part in parts:
        chunk = part.strip()
        if chunk.lower().startswith("filename*="):
            value = chunk.split("=", 1)[1].strip().strip('"')
            if "''" in value:
                value = value.split("''", 1)[1]
            return unquote(value)
        if chunk.lower().startswith("filename="):
            return chunk.split("=", 1)[1].strip().strip('"')
    return None


def _load_priority_metadata() -> dict[str, dict[str, Any]]:
    by_url: dict[str, dict[str, Any]] = {}
    for item in DOWNLOAD_PRIORITY:
        preferred = item.get("preferred")
        if isinstance(preferred, str) and preferred:
            by_url[preferred] = dict(item.get("metadata", {}))
    return by_url


def _build_sources(*, include_official: bool, include_all_user_sources: bool) -> list[SourceDef]:
    priority_meta_by_url = _load_priority_metadata()

    all_user = {
        key: SourceDef(
            key=key,
            url=meta["url"],
            source_group="user_provided",
            source_meta=meta,
            ingestion_metadata=priority_meta_by_url.get(meta["url"], {}),
        )
        for key, meta in USER_PROVIDED_URLS.items()
        if isinstance(meta.get("url"), str) and meta["url"]
    }

    if include_all_user_sources:
        sources = list(all_user.values())
    else:
        ordered_keys: list[str] = []
        for item in DOWNLOAD_PRIORITY:
            preferred = item.get("preferred")
            if not isinstance(preferred, str):
                continue
            match = next(
                (k for k, v in USER_PROVIDED_URLS.items() if v.get("url") == preferred), None
            )
            if match and match not in ordered_keys:
                ordered_keys.append(match)
        sources = [all_user[k] for k in ordered_keys if k in all_user]

    if include_official:
        for key, meta in OFFICIAL_SOURCES.items():
            url = meta.get("url")
            if not isinstance(url, str) or not url:
                continue
            sources.append(
                SourceDef(
                    key=key,
                    url=url,
                    source_group="official",
                    source_meta=meta,
                    ingestion_metadata=priority_meta_by_url.get(url, {}),
                )
            )

    deduped: dict[str, SourceDef] = {}
    for src in sources:
        deduped[src.key] = src
    return list(deduped.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_download(source_dir: Path) -> Path | None:
    for file in source_dir.iterdir() if source_dir.exists() else []:
        if not file.is_file():
            continue
        if file.name in {"metadata.json"}:
            continue
        return file
    return None


def _infer_doc_type(src: SourceDef) -> str:
    key = src.key.lower()
    text = f"{src.source_meta.get('content', '')} {src.source_meta.get('use_for', '')}".lower()

    if "circular" in key or "circular" in text:
        return "circular"
    if "notification" in key or "notification" in text:
        return "notification"
    if "rules" in key or "rules" in text:
        return "rules"
    if "act" in key or " act" in text:
        return "act"
    return ""


def _infer_act_name(src: SourceDef) -> str:
    key = src.key.lower()
    if "it_act_2025" in key:
        return "Income Tax Act 2025"
    if "it_act_1961" in key or "income_tax_act_1961" in key:
        return "Income Tax Act 1961"
    if "cgst" in key:
        return "CGST Act"
    if "igst" in key:
        return "IGST Act"
    if "utgst" in key:
        return "UTGST Act"
    if "income_tax_rules_2026" in key or "it_rules_2026" in key:
        return "Income Tax Rules 2026"
    return ""


def _build_chunk_metadata(src: SourceDef) -> dict[str, Any]:
    chunk_meta = dict(CHUNK_METADATA_TEMPLATE)
    chunk_meta["source_key"] = src.key
    chunk_meta["source_url"] = src.url

    # Fill from explicit ingest metadata first, then inferred defaults.
    for key, value in src.ingestion_metadata.items():
        if key in chunk_meta and isinstance(value, str):
            chunk_meta[key] = value

    if not chunk_meta["doc_type"]:
        chunk_meta["doc_type"] = _infer_doc_type(src)
    if not chunk_meta["act_name"]:
        chunk_meta["act_name"] = _infer_act_name(src)

    return chunk_meta


def _download_one(src: SourceDef) -> dict[str, Any]:
    source_dir = RAW_ROOT / src.key
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = source_dir / "metadata.json"
    chunk_metadata = _build_chunk_metadata(src)

    existing = _existing_download(source_dir)
    if existing:
        result = {
            "source_key": src.key,
            "url": src.url,
            "status": "skipped_existing",
            "source_group": src.source_group,
            "saved_path": str(existing.relative_to(ROOT).as_posix()),
            "saved_bytes": existing.stat().st_size,
            "saved_sha256": _sha256(existing),
            "checked_at": _now_iso(),
            "source_meta": src.source_meta,
            "ingestion_metadata": src.ingestion_metadata,
            "chunk_metadata": chunk_metadata,
        }
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    headers = {"User-Agent": USER_AGENT}
    try:
        with requests.get(src.url, headers=headers, timeout=TIMEOUT_SECONDS, stream=True) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            file_name = _filename_from_headers(resp)
            if not file_name:
                suffix = _safe_ext(content_type, src.url)
                file_name = f"source{suffix}"

            out_path = source_dir / file_name
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)

            size = out_path.stat().st_size
            result = {
                "source_key": src.key,
                "url": src.url,
                "status": "downloaded",
                "source_group": src.source_group,
                "http_status": resp.status_code,
                "content_type": content_type,
                "saved_path": str(out_path.relative_to(ROOT).as_posix()),
                "saved_bytes": size,
                "saved_sha256": _sha256(out_path),
                "downloaded_at": _now_iso(),
                "source_meta": src.source_meta,
                "ingestion_metadata": src.ingestion_metadata,
                "chunk_metadata": chunk_metadata,
            }
            metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    except Exception as exc:
        result = {
            "source_key": src.key,
            "url": src.url,
            "status": "failed",
            "source_group": src.source_group,
            "error": str(exc),
            "failed_at": _now_iso(),
            "source_meta": src.source_meta,
            "ingestion_metadata": src.ingestion_metadata,
            "chunk_metadata": chunk_metadata,
        }
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def download_sources(sources: list[SourceDef]) -> dict[str, Any]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    results = [_download_one(src) for src in sources]

    summary = {
        "generated_at": _now_iso(),
        "counts": {
            "total": len(results),
            "downloaded": sum(1 for r in results if r["status"] == "downloaded"),
            "skipped_existing": sum(1 for r in results if r["status"] == "skipped_existing"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
        },
        "results": results,
    }
    MANIFEST_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all USER_PROVIDED_URLS instead of only DOWNLOAD_PRIORITY set.",
    )
    parser.add_argument(
        "--include-official",
        action="store_true",
        help="Also include OFFICIAL_SOURCES links.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Download only specific source_key values (repeatable).",
    )
    args = parser.parse_args()

    sources = _build_sources(
        include_official=args.include_official,
        include_all_user_sources=args.all,
    )

    if args.source:
        wanted = set(args.source)
        sources = [s for s in sources if s.key in wanted]

    if not sources:
        print("No sources selected. Nothing to do.")
        return

    print(f"Selected {len(sources)} source(s).")
    summary = download_sources(sources)

    print("\nDownload summary")
    print("=" * 60)
    print(f"Total:     {summary['counts']['total']}")
    print(f"Downloaded:{summary['counts']['downloaded']}")
    print(f"Skipped:   {summary['counts']['skipped_existing']}")
    print(f"Failed:    {summary['counts']['failed']}")
    print(f"Manifest:  {MANIFEST_PATH.relative_to(ROOT).as_posix()}")

    failed = [r for r in summary["results"] if r["status"] == "failed"]
    if failed:
        print("\nFailures:")
        for row in failed:
            print(f"- {row['source_key']}: {row['error']}")


if __name__ == "__main__":
    main()
