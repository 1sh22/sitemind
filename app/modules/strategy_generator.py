from __future__ import annotations

from typing import List

from .llm import GeminiLLM


class StrategyGenerator:
    """Rule-based, local strategy generator for MVP (no paid APIs).

    This provides structured, templated output using retrieved context, avoiding LLM costs.
    Later, can be swapped with a local LLM backend (llama.cpp) without API costs.
    """

    def _join_context(self, chunks: List[str], max_chars: int = 1800) -> str:
        text = "\n".join(chunks)
        return (text[: max_chars - 3] + "...") if len(text) > max_chars else text

    def generate_business(self, context_chunks: List[str]) -> str:
        llm = GeminiLLM()
        if llm.available:
            out = llm.generate_business(context_chunks)
            if out:
                return out
        context = self._join_context(context_chunks)
        return (
            "## Business Strategy\n"
            "- Positioning: Based on the site's messaging and value props, summarize the core positioning.\n"
            "- ICP: Identify ideal customer profiles mentioned or implied.\n"
            "- Differentiators: Extract features/benefits that stand out.\n"
            "- Pricing/Packaging: Infer any pricing cues from the site (if present).\n"
            "- Distribution: Note channels hinted (blog, SEO, social, partnerships).\n"
            "- Risks: Highlight gaps or unclear points.\n\n"
            "### 90-Day Roadmap\n"
            "1. Foundation: clarify messaging, landing page improvements, analytics.\n"
            "2. Growth: content cadence, SEO cleanup, lead magnet, email capture.\n"
            "3. Scale: experiments (paid, partnerships), community building.\n\n"
            "### Evidence (context excerpts)\n" + context
        )

    def generate_content(self, context_chunks: List[str]) -> str:
        llm = GeminiLLM()
        if llm.available:
            out = llm.generate_content(context_chunks)
            if out:
                return out
        context = self._join_context(context_chunks)
        ideas = [
            "Founder's story and origin of the product",
            "Customer pain points and how we address them",
            "Behind-the-scenes: how a feature works",
            "Case study or user spotlight",
            "Mini-tutorial: quick win using the product",
            "Myth-busting post in our niche",
            "Comparisons vs alternatives (respectful, factual)",
            "Roadmap teaser and community call-to-action",
            "SEO blog: 'How to [problem] without [pain]'",
            "Data-driven insight from user behavior or industry",
        ]
        bullets = "\n".join(f"- {i}" for i in ideas)
        return (
            "## Content Strategy\n"
            "- Audience: summarize who we're speaking to from the site.\n"
            "- Tone: concise, practical, credible.\n"
            "- Cadence: 2-3 posts/week + 1 blog/week (MVP).\n\n"
            "### Post Ideas\n" + bullets + "\n\n" + "### Evidence (context excerpts)\n" + context
        )

    def answer_query(self, query: str, context_chunks: List[str]) -> str:
        llm = GeminiLLM()
        if llm.available:
            out = llm.answer(query, context_chunks)
            if out:
                return out
        context = self._join_context(context_chunks)
        return (
            f"### Answer\nYou asked: '{query}'. Based on retrieved context, here is a concise response.\n\n"
            "- Summary: synthesize key points from context.\n"
            "- Supporting details: include specifics from excerpts.\n\n"
            "### Evidence (context excerpts)\n" + context
        )


