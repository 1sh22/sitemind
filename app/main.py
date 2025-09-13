import os
import uuid
from typing import List, Optional

import streamlit as st

from modules.ingest import Ingest
from modules.parser import Parser
from modules.embeddings import Embeddings
from modules.retriever import Retriever
from modules.strategy_generator import StrategyGenerator
from modules.exporter import ProjectStore


def initialize_session_state() -> None:
    if "kb_index" not in st.session_state:
        st.session_state.kb_index = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts {role, content}
    if "project_id" not in st.session_state:
        st.session_state.project_id = str(uuid.uuid4())


def build_kb(url: str, max_pages: int = 8) -> None:
    ingest = Ingest(max_pages=max_pages)
    parser = Parser()
    embedder = Embeddings()
    retriever = Retriever()

    with st.spinner("Fetching website content..."):
        raw_texts = ingest.fetch_site(url)
    with st.spinner("Parsing and chunking content..."):
        chunks = parser.parse(raw_texts)
    with st.spinner("Embedding chunks and building index (first time downloads a small model)..."):
        index = embedder.build_index(chunks)

    st.session_state.kb_index = index
    st.session_state.chunks = chunks
    st.session_state.embedder = embedder
    st.session_state.retriever = retriever


def app_header() -> None:
    st.set_page_config(page_title="Sitemind", page_icon="🧠", layout="wide")
    _inject_styles()
    st.markdown(
        """
        <header class="sm-header">
          <div class="sm-brand"> SiteMind</div>
          
        </header>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls() -> None:
    st.sidebar.header("Project")
    st.sidebar.markdown(f"**ID**: `{st.session_state.project_id}`")
    if st.sidebar.button("New Project", type="secondary"):
        st.session_state.project_id = str(uuid.uuid4())
        st.session_state.kb_index = None
        st.session_state.chunks = []
        st.session_state.embedder = None
        st.session_state.retriever = None
        st.session_state.history = []
        st.rerun()


def strategy_section() -> None:
    st.markdown("<h3 class=sm-section>Strategy Generator</h3>", unsafe_allow_html=True)
    if st.session_state.kb_index is None:
        st.info("Build the knowledge base first.")
        return
    generator = StrategyGenerator()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Generate Business Strategy", type="primary"):
            with st.spinner("Generating business strategy..."):
                context_chunks = st.session_state.retriever.retrieve(
                    query="business overview positioning roadmap",  # seed query
                    index=st.session_state.kb_index,
                    embedder=st.session_state.embedder,
                    chunks=st.session_state.chunks,
                    top_k=8,
                )
                business = generator.generate_business(context_chunks)
                st.session_state.history.append({"role": "assistant", "content": business})
    with col_b:
        if st.button("Generate Content Strategy", type="primary"):
            with st.spinner("Generating content strategy..."):
                context_chunks = st.session_state.retriever.retrieve(
                    query="content ideas audience value props social blog",  # seed query
                    index=st.session_state.kb_index,
                    embedder=st.session_state.embedder,
                    chunks=st.session_state.chunks,
                    top_k=8,
                )
                content = generator.generate_content(context_chunks)
                st.session_state.history.append({"role": "assistant", "content": content})

    for message in st.session_state.history[-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def chat_section() -> None:
    st.markdown("<h3 class=sm-section>Chat over your KB</h3>", unsafe_allow_html=True)
    if st.session_state.kb_index is None:
        st.info("Build the knowledge base first.")
        return
    user_input = st.chat_input("Ask a question about the site or refine strategies...")
    if user_input:
        st.session_state.history.append({"role": "user", "content": user_input})
        top_chunks = st.session_state.retriever.retrieve(
            query=user_input,
            index=st.session_state.kb_index,
            embedder=st.session_state.embedder,
            chunks=st.session_state.chunks,
            top_k=6,
        )
        generator = StrategyGenerator()
        answer = generator.answer_query(user_input, top_chunks)
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.rerun()


def export_section() -> None:
    st.markdown("<h3 class=sm-section>Save / Export</h3>", unsafe_allow_html=True)
    store = ProjectStore()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Save Project", type="secondary"):
            data = {
                "project_id": st.session_state.project_id,
                "chunks": st.session_state.chunks,
                "history": st.session_state.history,
                "created_at": uuid.uuid1().hex,
            }
            saved_id = store.save_project(data)
            st.success(f"Saved with ID: {saved_id}")

    recent = store.list_projects(limit=8)
    if recent:
        opts = [rid for rid, _ in recent]
        load_id = st.selectbox("Load existing project", options=[""] + opts, index=0)
    else:
        load_id = st.text_input("Load by Project ID")
    if st.button("Load Project", type="secondary") and load_id:
        loaded = store.load_project(load_id)
        if loaded is None:
            st.error("Not found")
        else:
            st.session_state.project_id = loaded.get("project_id", load_id)
            st.session_state.chunks = loaded.get("chunks", [])
            st.session_state.history = loaded.get("history", [])
            if st.session_state.chunks:
                # rebuild index on load
                embedder = Embeddings()
                st.session_state.kb_index = embedder.build_index(st.session_state.chunks)
                st.session_state.embedder = embedder
                st.session_state.retriever = Retriever()
            st.success("Project loaded")

    with col2:
        if st.button("Export JSON", type="secondary"):
            export_path = store.export_json(
                {
                    "project_id": st.session_state.project_id,
                    "history": st.session_state.history,
                }
            )
            st.success(f"Exported to {export_path}")


def main() -> None:
    initialize_session_state()
    app_header()
    sidebar_controls()

    st.markdown("<h3 class=sm-section>Build Knowledge Base</h3>", unsafe_allow_html=True)
    url_col, btn_col = st.columns([7, 3], vertical_alignment="bottom")
    with url_col:
        url = st.text_input("Website URL", placeholder="https://example.com")
    with btn_col:
        ingest_clicked = st.button("Ingest & Build KB", type="primary", use_container_width=True)
    # Options row (compact)
    opt1, _ = st.columns([1, 9])
    with opt1:
        max_pages = st.number_input("Max pages", min_value=1, max_value=30, value=8, help="Limit crawl size")
    if ingest_clicked and url:
        try:
            build_kb(url, int(max_pages))
            if not st.session_state.chunks:
                st.warning("No readable content found. Try another page or increase pages.")
            else:
                st.success("Knowledge base built")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to build KB: {exc}")

    if st.session_state.kb_index is not None:
        st.success(f"KB ready with {len(st.session_state.chunks)} chunks")
        with st.container():
            strategy_section()
        with st.container():
            chat_section()
        with st.container():
            export_section()

    st.markdown(
        """
        <footer class="sm-footer">
            <div>© 2025 Sitemind • Minimal, responsive, local-first MVP</div>
            <div class="sm-links"><a href="#">Docs</a> · <a href="#">Feedback</a></div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            html, body, [class^="css"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
            /* Center page and limit width */
            .main .block-container { max-width: 900px; margin: 0 auto; }
            /* Header */
            .sm-header { position: sticky; top: 0; z-index: 10; backdrop-filter: blur(8px); border-bottom: 1px solid rgba(0,0,0,0.06); padding: 16px 6px; margin-bottom: 12px; display: flex; align-items: baseline; gap: 10px; justify-content: center; }
            .sm-brand { font-weight: 700; font-size: 30px; letter-spacing: -0.02em; }
            .sm-sub { color: #6b7280; font-size: 13px; }
            .sm-section { margin: 18px 0 10px 0; padding-bottom: 6px; border-bottom: 1px dashed rgba(0,0,0,0.07); }
            /* Minimal white buttons */
            div.stButton>button, button[kind] { background: #fff !important; color: #111 !important; border: 1px solid rgba(0,0,0,0.12) !important; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); transition: transform 120ms ease, box-shadow 120ms ease; }
            div.stButton>button:hover, button[kind]:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(0,0,0,0.06); }
            .stChatFloatingInputContainer { border-top: 1px solid rgba(0,0,0,0.06); }
            .sm-footer { margin-top: 28px; padding: 12px 6px; color: #6b7280; font-size: 12px; border-top: 1px solid rgba(0,0,0,0.06); display:flex; justify-content: space-between; }
            .sm-links a { color: inherit; text-decoration: none; }
            .sm-links a:hover { text-decoration: underline; }
            /* Response headings slightly smaller */
            .main h1 { font-size: 1.4rem; }
            .main h2 { font-size: 1.2rem; }
            .main h3 { font-size: 1.05rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


