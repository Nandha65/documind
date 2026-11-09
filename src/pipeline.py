import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Resolve paths from the project root so the app works from any working directory
DATA_DIR = str(PROJECT_ROOT / "data")
DB_DIR = str(PROJECT_ROOT / "chroma_db")
# Content hash of data/ that each collection was built from
INDEX_STATE_FILE = Path(DB_DIR) / "index_state.json"

# Retrieve more chunks than LlamaIndex's default of 2, so answers that draw on
# several sections or documents get the context they need.
SIMILARITY_TOP_K = 4
# Recent chat messages sent with each question so follow-ups keep their context
MAX_HISTORY_MESSAGES = 6


def _data_files() -> list[Path]:
    """The files SimpleDirectoryReader loads: top-level, non-hidden files in data/."""
    root = Path(DATA_DIR)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and not p.name.startswith("."))


def data_signature() -> tuple:
    """Cheap snapshot of data/ (name, size, modified time), used as a cache key by the app."""
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in _data_files())


def data_fingerprint() -> str:
    """Content hash of data/. Changes whenever a file is added, removed or edited."""
    digest = hashlib.sha256()
    for path in _data_files():
        digest.update(path.name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _stored_fingerprint(collection_name: str) -> str | None:
    try:
        return json.loads(INDEX_STATE_FILE.read_text()).get(collection_name)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _store_fingerprint(collection_name: str, fingerprint: str) -> None:
    try:
        state = json.loads(INDEX_STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state[collection_name] = fingerprint
    INDEX_STATE_FILE.write_text(json.dumps(state, indent=2))

def configure_provider(provider: str):
    """Configures LLM and Embedding models based on selected provider."""
    # Re-read .env so keys added while the app is running are picked up
    load_dotenv(dotenv_path=env_path)
    provider = provider.lower()

    if provider == "gemini":
        from google.genai import types
        from llama_index.llms.google_genai import GoogleGenAI
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")

        # Gemini 3 models default to a high thinking level, which can add several
        # seconds before any answer text is produced. Answering from retrieved
        # chunks doesn't need deep reasoning, so keep thinking low.
        llm = GoogleGenAI(
            model="gemini-3.6-flash",
            api_key=api_key,
            generation_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-2-preview", api_key=api_key)
        collection_name = "documind_gemini"

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file.")

        llm = OpenAI(model="gpt-4o-mini", api_key=api_key)
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
        collection_name = "documind_openai"

    elif provider == "claude":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        from .claude_llm import ClaudeLLM

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file.")

        llm = ClaudeLLM(api_key=api_key)
        # Anthropic has no embeddings API, so embed locally. This also avoids a
        # network round trip for every question (~2 ms instead of ~300 ms).
        embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=str(PROJECT_ROOT / ".fastembed_cache"),
        )
        collection_name = "documind_claude"

    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose 'gemini', 'openai' or 'claude'.")

    return llm, embed_model, collection_name

def initialize_index(provider: str = "gemini"):
    """Builds or loads an index into a provider-specific Chroma collection."""
    llm, embed_model, collection_name = configure_provider(provider)
    fingerprint = data_fingerprint()

    # 1. Connect to local ChromaDB
    db = chromadb.PersistentClient(path=DB_DIR)
    chroma_collection = db.get_or_create_collection(collection_name)

    # 2. Reuse existing embeddings if they were built from the current data/
    if chroma_collection.count() > 0 and _stored_fingerprint(collection_name) == fingerprint:
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        return index, llm

    # 3. Check source documents
    if not _data_files():
        raise ValueError(f"Please add source documents to '{DATA_DIR}' before running.")

    # 4. First run or data/ changed: rebuild this provider's collection from scratch
    db.delete_collection(collection_name)
    chroma_collection = db.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    _store_fingerprint(collection_name, fingerprint)
    return index, llm

class FollowUpChatEngine(CondensePlusContextChatEngine):
    """Chat engine that searches with the previous question plus the new one.

    LlamaIndex normally asks the LLM to rewrite each follow-up as a standalone
    question before searching, which adds an LLM call to every question after
    the first. Searching with the previous question prepended finds the right
    documents for follow-ups like "who do I ask about it?" at no extra cost.
    The LLM still answers the question as asked, with the conversation in view.
    """

    def _condense_question(self, chat_history: list[ChatMessage], latest_message: str) -> str:
        previous = next(
            (m.content for m in reversed(chat_history) if m.role == MessageRole.USER), None
        )
        return f"{previous}\n{latest_message}" if previous else latest_message

    async def _acondense_question(self, chat_history: list[ChatMessage], latest_message: str) -> str:
        return self._condense_question(chat_history, latest_message)


def stream_answer(index, llm, question: str, history: list[dict]):
    """Answer a question in the context of the conversation so far.

    history holds earlier {"role", "content"} messages. Returns a streaming
    chat response with .response_gen (answer tokens) and .source_nodes.
    """
    engine = FollowUpChatEngine.from_defaults(
        retriever=index.as_retriever(similarity_top_k=SIMILARITY_TOP_K),
        llm=llm,
    )
    chat_history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in history[-MAX_HISTORY_MESSAGES:]
    ]
    return engine.stream_chat(question, chat_history=chat_history)


def format_sources(source_nodes, excerpt_chars: int = 240) -> list[dict]:
    """One entry per source document (and PDF page) behind an answer, best match first."""
    sources, seen = [], set()
    for node_with_score in source_nodes:
        meta = node_with_score.node.metadata
        key = (meta.get("file_name", "unknown"), meta.get("page_label"))
        if key in seen:
            continue
        seen.add(key)
        text = " ".join(node_with_score.node.get_content().split())
        sources.append({
            "file": key[0],
            "page": key[1],
            "score": node_with_score.score,
            "excerpt": text[:excerpt_chars] + ("…" if len(text) > excerpt_chars else ""),
        })
    return sources


def get_query_engine(provider: str = "gemini"):
    index, llm = initialize_index(provider=provider)
    # Stream tokens as they are generated so the answer starts appearing right
    # away instead of only after the whole response has been produced.
    return index.as_query_engine(llm=llm, streaming=True, similarity_top_k=SIMILARITY_TOP_K)

if __name__ == "__main__":
    engine = get_query_engine("gemini")
    response = engine.query("What is this document about?")
    print("\n--- Gemini Query Test Result ---")
    response.print_response_stream()
    print()