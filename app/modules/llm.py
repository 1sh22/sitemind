from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv


class GeminiLLM:
    """Thin wrapper around Google's Gemini API.

    Uses free-tier if available. If no API key present, acts as unavailable.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self._client = None
        if self.api_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate_business(self, context_chunks: List[str]) -> str:
        prompt = self._business_prompt(context_chunks)
        return self._generate(prompt)

    def generate_content(self, context_chunks: List[str]) -> str:
        prompt = self._content_prompt(context_chunks)
        return self._generate(prompt)

    def answer(self, query: str, context_chunks: List[str]) -> str:
        prompt = self._qa_prompt(query, context_chunks)
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        if not self.available:
            return ""
        try:
            result = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 1200,
                },
            )
            return (result.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            return f"[Gemini error] {exc}"

    def _join(self, chunks: List[str], max_chars: int = 3000) -> str:
        text = "\n".join(chunks)
        return (text[: max_chars - 3] + "...") if len(text) > max_chars else text

    def _business_prompt(self, chunks: List[str]) -> str:
        context = self._join(chunks)
        return (
            "You are a senior GTM strategist. Using ONLY the provided website context, "
            "produce a concise, well-structured business strategy in markdown.\n\n"
            "Required sections with clear bullets and short sentences:\n"
            "1. Positioning\n"
            "2. Ideal Customer Profiles\n"
            "3. Differentiators\n"
            "4. Pricing & Packaging (if evidence)\n"
            "5. Distribution Channels\n"
            "6. Risks & Gaps\n"
            "7. 90-Day Roadmap (3 phases)\n\n"
            "Constraints:\n- Be factual; cite evidence only from context.\n- Keep it under 450 words.\n\n"
            f"Context:\n'''\n{context}\n'''\n"
        )

    def _content_prompt(self, chunks: List[str]) -> str:
        context = self._join(chunks)
        return (
            "You are a content strategist. Based ONLY on the website context, "
            "create a concise content plan in markdown.\n\n"
            "Sections:\n"
            "1. Audience & Tone\n"
            "2. Content Pillars (3-5)\n"
            "3. Posting Cadence\n"
            "4. 10 Post Ideas (one-liners)\n"
            "5. One Sample LinkedIn Post (<=120 words)\n\n"
            "Constraints:\n- Be factual to the context.\n- Crisp bullets.\n\n"
            f"Context:\n'''\n{context}\n'''\n"
        )

    def _qa_prompt(self, query: str, chunks: List[str]) -> str:
        context = self._join(chunks)
        return (
            "Answer the user's question grounded strictly in the provided context. "
            "Return markdown with: Summary, Supporting Details, and a short Next Step.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n'''\n{context}\n'''\n"
        )


