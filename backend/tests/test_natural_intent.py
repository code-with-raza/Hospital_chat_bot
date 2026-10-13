import json
from types import SimpleNamespace

from app.graph import nodes
from app.graph.graph import _knowledge_module, build_graph, route


def test_ambiguous_classification_uses_safe_recent_context(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content=json.dumps({
                "intent": "general_hospital_query",
                "resolved_question": "What are the ICU visitor limits?",
            }))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(nodes, "LLM_API_KEY", "test-key")
    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    history = [
        {"role": "user", "content": "What are the ICU visiting hours?"},
        {"role": "assistant", "content": "They are 5 PM to 6 PM daily."},
        {"role": "user", "content": "Check appointment PAT001, phone 03001234567."},
    ]

    result = nodes.understand_with_llm("Would that still apply later?", history)
    prompt = captured["messages"][1]["content"]

    assert result["intent"] == "general_hospital_query"
    assert result["resolved_question"] == "What are the ICU visitor limits?"
    assert "ICU visiting hours" in prompt
    assert "Would that still apply later?" in prompt
    assert "PAT001" not in prompt
    assert "03001234567" not in prompt


def test_contextual_paraphrase_routes_to_rag_without_interrupting_booking(monkeypatch):
    requested = []
    monkeypatch.setattr(
        nodes,
        "understand_with_llm",
        lambda message, history: {
            "intent": "general_hospital_query",
            "resolved_question": "What are the ICU visitor limits after dark?",
        },
    )
    monkeypatch.setattr(
        _knowledge_module,
        "answer_question",
        lambda question: (
            requested.append(question)
            or {"answer": "Please check the hospital policy for current information.", "sources": []}
        ),
    )
    graph = build_graph()
    history = [
        {"role": "user", "content": "What are the ICU visiting hours?"},
        {"role": "assistant", "content": "They are 5 PM to 6 PM daily."},
    ]
    config = {"configurable": {"thread_id": "context-paraphrase-booking"}}
    result = graph.invoke(
        {
            "user_message": "Would that still apply later?",
            "conversation_history": history,
            "pending_action": "date",
            "appointment_date": "2027-01-12",
        },
        config=config,
    )

    saved = graph.get_state(config).values
    assert result["intent"] == "general_hospital_query"
    assert route(result) == "knowledge"
    assert requested == ["What are the ICU visitor limits after dark?"]
    assert saved["pending_action"] == "date"
    assert saved["appointment_date"] == "2027-01-12"


def test_ambiguous_acknowledgment_uses_conversation_intent(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "understand_with_llm",
        lambda message, history: {"intent": "general_chat"},
    )
    graph = build_graph()
    result = graph.invoke(
        {
            "user_message": "Fair enough",
            "conversation_history": [
                {"role": "assistant", "content": "I can help with hospital information."}
            ],
        },
        config={"configurable": {"thread_id": "natural-ack"}},
    )

    assert result["intent"] == "general_chat"
    assert result["response_type"] == "text"
    assert "didn't quite understand" not in result["response"].casefold()
