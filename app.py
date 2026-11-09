import logging
import re

import streamlit as st
from src.errors import friendly_error
from src.pipeline import data_signature, format_sources, initialize_index, stream_answer

logger = logging.getLogger(__name__)

# Characters Streamlit would treat as Markdown (or LaTeX, for "$")
_MARKDOWN_CHARS = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$])")

st.set_page_config(page_title="DocuMind", page_icon="🤖", layout="wide")

st.title("🤖 DocuMind: Multi-Provider RAG Assistant")
st.write("Query your local documents with **Google Gemini**, **OpenAI** or **Anthropic Claude**.")

PROVIDERS = {"Google Gemini": "gemini", "OpenAI": "openai", "Anthropic Claude": "claude"}

# Sidebar Provider Selection
st.sidebar.header("Model Settings")
provider_choice = st.sidebar.selectbox(
    "Select AI Provider:",
    tuple(PROVIDERS),
    index=0
)
provider_key = PROVIDERS[provider_choice]

# Cache the index per provider. Exceptions are not cached, so fixing a
# configuration problem (e.g. a missing API key) takes effect on the next rerun.
# data_version is only part of the cache key: when files in data/ change, the
# index is reloaded, and the pipeline re-indexes if their content changed.
@st.cache_resource(show_spinner=False, max_entries=3)
def load_index(provider: str, data_version: tuple):
    return initialize_index(provider)

try:
    with st.spinner(f"Loading {provider_choice} index & embeddings..."):
        index, llm = load_index(provider_key, data_signature())
except Exception as e:
    st.error(f"⚠️ Error initializing {provider_choice}: {e}")
    st.stop()


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            label = source["file"] + (f", page {source['page']}" if source["page"] else "")
            score = f" · relevance {source['score']:.2f}" if source["score"] is not None else ""
            st.markdown(f"**{_MARKDOWN_CHARS.sub(r'\\\1', label)}**{score}")
            # Excerpts are raw document text; show them as-is, not as Markdown
            st.caption(_MARKDOWN_CHARS.sub(r"\\\1", source["excerpt"]))


# Reset chat when switching provider
if "current_provider" not in st.session_state or st.session_state.current_provider != provider_key:
    st.session_state.current_provider = provider_key
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    history = list(st.session_state.messages)
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with st.spinner(f"Querying via {provider_choice}..."):
                response = stream_answer(index, llm, prompt, history)
            # Render tokens as they arrive instead of waiting for the full answer
            answer = st.write_stream(response.response_gen)
            sources = format_sources(response.source_nodes)
            show_sources(sources)
        except Exception as e:
            # Rate limits, overloaded models and network errors: explain instead
            # of showing a traceback. The full error still goes to the terminal.
            logger.exception("Query via %s failed", provider_choice)
            st.error(f"⚠️ {friendly_error(provider_choice, e)}")
            answer = None

    if answer is None:
        # Drop the unanswered question so it isn't sent as history next time
        st.session_state.messages.pop()
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
