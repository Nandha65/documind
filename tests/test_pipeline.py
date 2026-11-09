"""Index building, data change detection and source formatting (no network)."""
import chromadb
import pytest
from llama_index.core import Document, MockEmbedding, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole, MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

from src import pipeline


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("The meal allowance is $75 per day.")
    (data / "b.txt").write_text("Office days are Tuesdays and Thursdays.")
    monkeypatch.setattr(pipeline, "DATA_DIR", str(data))
    return data


@pytest.fixture
def offline_provider(tmp_path, monkeypatch, data_dir):
    """Index into a temporary Chroma DB with fake embeddings."""
    db_dir = tmp_path / "chroma_db"
    monkeypatch.setattr(pipeline, "DB_DIR", str(db_dir))
    monkeypatch.setattr(pipeline, "INDEX_STATE_FILE", db_dir / "index_state.json")
    monkeypatch.setattr(
        pipeline,
        "configure_provider",
        lambda provider: (MockLLM(), MockEmbedding(embed_dim=8), "test_collection"),
    )
    return data_dir


def test_fingerprint_tracks_content_and_ignores_hidden_files(data_dir):
    before = pipeline.data_fingerprint()
    (data_dir / ".DS_Store").write_bytes(b"\0")
    assert pipeline.data_fingerprint() == before
    (data_dir / "a.txt").write_text("The meal allowance is $80 per day.")
    assert pipeline.data_fingerprint() != before


def test_signature_changes_when_a_file_is_added(data_dir):
    before = pipeline.data_signature()
    (data_dir / "c.txt").write_text("A new document.")
    assert pipeline.data_signature() != before


def test_index_is_reused_until_data_changes(offline_provider, monkeypatch):
    loads = []
    real_reader = pipeline.SimpleDirectoryReader

    def counting_reader(*args, **kwargs):
        loads.append(1)
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(pipeline, "SimpleDirectoryReader", counting_reader)

    pipeline.initialize_index("test")
    pipeline.initialize_index("test")
    assert len(loads) == 1  # the second load reused the stored index

    (offline_provider / "c.txt").write_text("A new document.")
    pipeline.initialize_index("test")
    assert len(loads) == 2  # data changed, so it rebuilt

    collection = chromadb.PersistentClient(path=pipeline.DB_DIR).get_collection("test_collection")
    assert collection.count() == 3  # rebuilt from scratch: one chunk per file, no duplicates


def test_missing_documents_raise_a_clear_error(offline_provider):
    for path in offline_provider.iterdir():
        path.unlink()
    with pytest.raises(ValueError, match="add source documents"):
        pipeline.initialize_index("test")


def test_follow_up_search_includes_previous_question():
    index = VectorStoreIndex.from_documents([Document(text="hello")], embed_model=MockEmbedding(embed_dim=8))
    engine = pipeline.FollowUpChatEngine.from_defaults(retriever=index.as_retriever(), llm=MockLLM())
    history = [
        ChatMessage(role=MessageRole.USER, content="What is the meal allowance?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="$75 per day."),
    ]
    assert engine._condense_question(history, "Who do I ask about it?") == (
        "What is the meal allowance?\nWho do I ask about it?"
    )
    assert engine._condense_question([], "First question") == "First question"


def test_format_sources_one_entry_per_file_and_page():
    nodes = [
        NodeWithScore(node=TextNode(text="Key result 1.1:   3,800 customers", metadata={"file_name": "okrs.pdf", "page_label": "1"}), score=0.8),
        NodeWithScore(node=TextNode(text="More from the same page", metadata={"file_name": "okrs.pdf", "page_label": "1"}), score=0.6),
        NodeWithScore(node=TextNode(text="x" * 500, metadata={"file_name": "handbook.md"}), score=0.5),
    ]
    sources = pipeline.format_sources(nodes)
    assert [(s["file"], s["page"]) for s in sources] == [("okrs.pdf", "1"), ("handbook.md", None)]
    assert sources[0]["excerpt"] == "Key result 1.1: 3,800 customers"
    assert sources[1]["excerpt"] == "x" * 240 + "…"
