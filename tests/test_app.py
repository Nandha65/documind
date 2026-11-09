"""The Streamlit app, with index loading and answering stubbed out."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.test_errors import GEMINI_OVERLOADED

APP = str(Path(__file__).resolve().parent.parent / "app.py")
QUESTION = "What is the meal allowance?"


def run_app(stream_answer):
    with patch("src.pipeline.initialize_index", return_value=(object(), object())), \
         patch("src.pipeline.stream_answer", stream_answer):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.chat_input[0].set_value(QUESTION).run()
    return app


def test_answer_is_streamed_and_saved_to_history():
    response = SimpleNamespace(response_gen=iter(["$75 ", "per day."]), source_nodes=[])
    app = run_app(lambda *args: response)
    assert not app.error
    assert [(m["role"], m["content"]) for m in app.session_state.messages] == [
        ("user", QUESTION),
        ("assistant", "$75 per day."),
    ]


def test_overloaded_provider_shows_a_friendly_error():
    def overloaded(*args):
        raise GEMINI_OVERLOADED

    app = run_app(overloaded)
    assert "overloaded right now" in app.error[0].value
    assert not app.exception
    assert app.session_state.messages == []  # the unanswered question is dropped


def test_error_part_way_through_an_answer():
    def broken_stream():
        yield "Partial "
        raise ConnectionError("connection reset by peer")

    app = run_app(lambda *args: SimpleNamespace(response_gen=broken_stream(), source_nodes=[]))
    assert "Couldn't reach" in app.error[0].value
    assert app.session_state.messages == []
