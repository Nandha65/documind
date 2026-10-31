import streamlit as st
from src.pipeline import get_query_engine

st.set_page_config(page_title="DocuMind", page_icon="🤖", layout="wide")

st.title("🤖 DocuMind: Multi-Provider RAG Assistant")
st.write("Query your local documents with either **Google Gemini** or **OpenAI**.")

# Sidebar Provider Selection
st.sidebar.header("Model Settings")
provider_choice = st.sidebar.selectbox(
    "Select AI Provider:",
    ("Google Gemini", "OpenAI"),
    index=0
)
provider_key = "gemini" if provider_choice == "Google Gemini" else "openai"

# Cache the engine per provider
@st.cache_resource(show_spinner=False)
def load_engine(provider: str):
    try:
        return get_query_engine(provider)
    except Exception as e:
        return str(e)

with st.spinner(f"Loading {provider_choice} index & embeddings..."):
    engine_or_error = load_engine(provider_key)

if isinstance(engine_or_error, str):
    st.error(f"⚠️ Error initializing {provider_choice}: {engine_or_error}")
else:
    query_engine = engine_or_error

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
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": str(response)})