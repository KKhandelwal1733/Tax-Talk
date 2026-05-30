from __future__ import annotations

from tax_talk.core.config import settings
from tax_talk.ingestion.chunking_strategies.fixed import FixedChunkingStrategy
from tax_talk.ingestion.chunking_strategies.helpers import prepend_contextual_summary
from tax_talk.ingestion.contextual_summary import ContextualSummaryResult, build_contextual_summary
from tax_talk.ingestion.loader import SourceDocument


class ContextualChunkingStrategy(FixedChunkingStrategy):
    """Fixed-size chunking plus contextual summary prefixing."""

    name = "contextual"

    def build_summary(self, doc: SourceDocument, cleaned_text: str) -> ContextualSummaryResult:
        return build_contextual_summary(doc, cleaned_text)

    def render_chunk(self, chunk_text: str, summary: ContextualSummaryResult) -> str:
        return prepend_contextual_summary(
            chunk_text,
            summary.text,
            label=settings.contextual_summary_prefix_label,
        )
