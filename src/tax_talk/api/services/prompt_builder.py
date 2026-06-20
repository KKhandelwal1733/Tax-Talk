"""Prompt construction for grounded chat answers."""

from __future__ import annotations

from typing import Any


def build_chat_prompt(*, query: str, hits: list[dict[str, Any]]) -> str:
    """Build a grounded prompt that includes retrieval context with metadata."""
    if not hits:
        return (
            "You are an expert Tax AI Assistant specialized in Indian Tax Statutes "
            "(CGST Act 2017, IGST Act 2017, Income-tax Act 1961, and Income-tax Act 2025) "
            "and CBIC circulars, notifications, and GST Council decisions.\n"
            "The provided context is empty, so do not fabricate legal citations.\n"
            "State uncertainty clearly and explain that the available context is insufficient.\n\n"
            f"Question: {query}\n\n"
            "Context Chunks:\n[none]\n\n"
            "Answer:"
        )

    context_rows: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        text = str(hit.get("text", "")).strip()
        source_key = str(hit.get("source_key", "unknown"))
        section_title = str(hit.get("section_title", ""))
        chunk_id = str(hit.get("chunk_id", ""))
        if not text:
            continue
        context_rows.append(
            f"[{idx}] source={source_key} section={section_title} chunk_id={chunk_id}\n{text}"
        )

    context_block = "\n\n".join(context_rows)
    return (
        "You are an expert Tax AI Assistant specialized in Indian Tax Statutes "
        "(CGST Act 2017, IGST Act 2017, Income-tax Act 1961, and Income-tax Act 2025) "
        "and CBIC circulars, notifications, and GST Council decisions.\n"
        "Your task is to answer the following user question by analyzing and synthesizing the provided tax context chunks.\n\n"
        "CRITICAL CONSTRAINTS & REASONING GUIDELINES:\n"
        "1. SYNTHESIZE: Read the provided context fragments collectively. If the complete answer requires "
        "combining information from multiple chunks (e.g., a general rule in one chunk and an exception or "
        "threshold in another), combine them into a single coherent legal conclusion. Do not treat any single "
        "chunk as the whole answer.\n"
        "2. AMENDMENTS FIRST: Each chunk is labelled with its source, document type, applicable period, and "
        "section reference where available. If a chunk is marked as 'amended' or carries a later applicable "
        "period than another chunk covering the same provision, prefer the amended version and explicitly note "
        "the change with its effective date.\n"
        "3. CIRCULAR vs. STATUTE: Circulars and notifications clarify but do not override statute. Where both "
        "are present, state the statutory basis first and then the clarification from the circular.\n"
        "4. STATUTORY SYNONYMS: Do not return a negative answer due to terminology mismatch. Recognize "
        "equivalent terms across statutes, for example: 'clinical establishment' or 'authorized medical "
        "practitioner' maps to healthcare or hospital context; 'renting of immovable property' maps to "
        "leasing or letting out property; 'consideration' maps to price or payment for supply.\n"
        "5. TEMPORAL FLAGS: If the answer applies only from a specific date, financial year, or notification "
        "effective date, state this explicitly in your response.\n"
        "6. MANDATORY GROUNDING: Rely strictly on the facts present in the provided context. Do not invent "
        "or assume section numbers, notification numbers, rates, thresholds, or dates not present in the text. "
        "If the context is genuinely insufficient to derive a reliable legal conclusion, state clearly: "
        "'The provided context does not specify [specific missing information].'\n\n"
        "OUTPUT FORMAT:\n"
        "- Begin with a direct answer to the question in 1 to 3 sentences.\n"
        "- Follow with supporting detail in bullet points if the answer involves multiple conditions, rates, "
        "thresholds, or exceptions. Use prose for simple single-rule answers.\n"
        "- End with a 'Legal Basis:' line citing the specific sections, notifications, or circulars referenced "
        "in your answer, using the format: Section X, [Act Name] [Year]; Circular No., date.\n"
        "- Keep the total response under 300 words unless the question requires a detailed slab or rate structure.\n\n"
        f"Question: {query}\n\n"
        "Context Chunks:\n"
        f"{context_block}\n\n"
        "Answer:"
    )
