"""
Generation layer: grounded answer synthesis via OpenAI GPT-4o-mini.
Returns None silently when OPENAI_API_KEY is not set.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a temporal question-answering assistant. Answer using ONLY the "
    "provided passages. Each passage lists the years mentioned in its text — "
    "use these to reason about time. If facts changed over time, explicitly "
    "state when each version was true, citing passage numbers. If the passages "
    "do not contain sufficient information, say exactly: 'The retrieved passages "
    "do not contain enough information to answer this question.' Never use "
    "knowledge outside the provided passages."
)


def generate_answer(
    query: str,
    reranked_passages: list,
    target_year: int,
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import openai

        context_parts = []
        for i, p in enumerate(reranked_passages[:5], start=1):
            years = p.get("years_in_text", [])
            years_str = ", ".join(str(y) for y in years) if years else "unknown"
            context_parts.append(
                f"[Passage {i} | Years mentioned: {years_str}]\n{p['text']}"
            )
        context = "\n\n".join(context_parts)

        user_msg = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"As of year: {target_year}\n\n"
            "Answer (cite passage numbers):"
        )

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return response.choices[0].message.content

    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return None
