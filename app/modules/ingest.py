from __future__ import annotations

import re
from collections import deque
from typing import Iterable, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential


class Ingest:
    """Simple breadth-first crawler that collects text from a site within same domain.

    Limits total pages and depth to stay lightweight for MVP.
    """

    def __init__(self, max_pages: int = 10, max_depth: int = 2, timeout_seconds: int = 15) -> None:
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout_seconds = timeout_seconds

    def fetch_site(self, start_url: str) -> List[str]:
        normalized = self._normalize_url(start_url)
        domain = urlparse(normalized).netloc
        visited: Set[str] = set()
        queue: deque[tuple[str, int]] = deque([(normalized, 0)])
        collected_texts: List[str] = []

        while queue and len(visited) < self.max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > self.max_depth:
                continue
            try:
                html = self._fetch_html(url)
            except Exception:
                visited.add(url)
                continue

            visited.add(url)
            texts, links = self._extract_text_and_links(html, base_url=url)
            collected_texts.extend(texts)

            for link in links:
                if self._is_same_domain(link, domain) and link not in visited:
                    queue.append((link, depth + 1))

        # Fallback if page has little body text: include title and meta description
        filtered = [t for t in collected_texts if t and t.strip()]
        if not filtered and html:
            soup = BeautifulSoup(html, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""
            desc = ""
            meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if meta and meta.get("content"):
                desc = str(meta.get("content")).strip()
            fallback = "\n".join(x for x in [title, desc] if x)
            if fallback:
                filtered = [fallback]
        return filtered

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _fetch_html(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def _extract_text_and_links(self, html: str, base_url: str) -> tuple[List[str], List[str]]:
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/nav/footer elements
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
            tag.decompose()

        texts: List[str] = []
        # Prefer headings and paragraphs and list items
        for selector in ["h1", "h2", "h3", "p", "li"]:
            for node in soup.select(selector):
                text = self._clean_text(node.get_text(separator=" "))
                if text:
                    texts.append(text)

        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href or href.startswith("#"):
                continue
            if href.startswith("mailto:") or href.startswith("tel:"):
                continue
            absolute = urljoin(base_url, href)
            links.append(absolute)

        return texts, links

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 40:  # drop very short items
            return ""
        return text

    def _normalize_url(self, url: str) -> str:
        # Strip leading '@' (e.g., inputs like @https://example.com)
        if url.startswith("@"):
            url = url[1:]
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _is_same_domain(self, url: str, domain: str) -> bool:
        return urlparse(url).netloc == domain


