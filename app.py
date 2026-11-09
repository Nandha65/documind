import streamlit as st
from src.pipeline import data_signature, get_query_engine

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

# Cache the engine per provider. Exceptions are not cached, so fixing a
# configuration problem (e.g. a missing API key) takes effect on the next rerun.
# data_version is only part of the cache key: when files in data/ change, a new
# engine is loaded, and the pipeline re-indexes if their content changed.
@st.cache_resource(show_spinner=False, max_entries=3)
def load_engine(provider: str, data_version: tuple):
    return get_query_engine(provider)

try:
    with st.spinner(f"Loading {provider_choice} index & embeddings..."):
        query_engine = load_engine(provider_key, data_signature())
except Exception as e:
    st.error(f"⚠️ Error initializing {provider_choice}: {e}")
    st.stop()

# Reset chat when switching provider
if "current_provider" not in st.session_state or st.session_state.current_provider != provider_key:
    st.session_state.current_provider = provider_key
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner(f"Querying via {provider_choice}..."):
            response = query_engine.query(prompt)
        # Render tokens as they arrive instead of waiting for the full answer
        answer = st.write_stream(response.response_gen)

    st.session_state.messages.append({"role": "assistant", "content": answer})
