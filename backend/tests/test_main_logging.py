import logging

from app import main
from app.schemas.chat import ChatRequest


def test_chat_logs_do_not_include_user_message_or_patient_values(monkeypatch, caplog):
    class GraphStub:
        @staticmethod
        def get_state(config):
            return type("Snapshot", (), {"values": {}})()

        @staticmethod
        def invoke(*args, **kwargs):
            return {"response": "Please verify your record.", "response_type": "text"}

        @staticmethod
        def update_state(*args, **kwargs):
            return None

    monkeypatch.setattr(main, "chat_graph", GraphStub())
    sensitive_message = "Ali Raza 03001234567 back pain PAT001"
    with caplog.at_level(logging.INFO):
        main.chat(ChatRequest(session_id="test-session", message=sensitive_message))

    assert sensitive_message not in caplog.text
    assert "03001234567" not in caplog.text
    assert "back pain" not in caplog.text


def test_chat_error_log_omits_sensitive_exception_text(monkeypatch, caplog):
    class GraphStub:
        @staticmethod
        def get_state(config):
            return type("Snapshot", (), {"values": {}})()

        @staticmethod
        def invoke(*args, **kwargs):
            raise RuntimeError("failed for Ali Raza 03001234567 with back pain")

        @staticmethod
        def update_state(*args, **kwargs):
            return None

    monkeypatch.setattr(main, "chat_graph", GraphStub())
    with caplog.at_level(logging.ERROR):
        result = main.chat(ChatRequest(
            session_id="test-session",
            message="PAT001 03001234567 back pain",
        ))

    assert result.type == "error"
    assert "RuntimeError" in caplog.text
    assert "03001234567" not in caplog.text
    assert "Ali Raza" not in caplog.text
    assert "back pain" not in caplog.text

