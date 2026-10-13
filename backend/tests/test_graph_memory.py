import pytest

from app.graph import nodes
from app.graph.graph import _knowledge_module, build_graph, route


def _invoke_with_history(graph, config, message):
    history = list(graph.get_state(config).values.get("conversation_history", []))
    result = graph.invoke(
        {"user_message": message, "conversation_history": history},
        config=config,
    )
    graph.update_state(config, {"conversation_history": [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.get("response", "")},
    ][-12:]})
    return result


def test_checkpointed_followup_state_is_isolated_by_session(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    graph = build_graph()
    first = {"configurable": {"thread_id": "test-session-first"}}
    second = {"configurable": {"thread_id": "test-session-second"}}

    graph.invoke({"user_message": "Check my appointment"}, config=first)
    graph.invoke({"user_message": "hello"}, config=second)
    first_state = graph.get_state(first).values
    second_state = graph.get_state(second).values

    assert first_state["pending_action"] == "lookup_appointment"
    assert second_state["intent"] == "general_chat"
    assert second_state.get("pending_action") is None
    assert "messages" not in first_state
    assert "messages" not in second_state

    graph.invoke({"user_message": "PAT001"}, config=first)
    resumed = graph.get_state(first).values
    assert resumed["pending_action"] == "lookup_appointment"
    assert resumed["lookup_patient_id"] == "PAT001"
    assert graph.get_state(second).values.get("pending_action") is None


def test_explicit_new_booking_interrupts_lookup_and_clears_lookup_fields(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    pending = {
        "user_message": "I want to book an appointment with a cardiologist",
        "pending_action": "lookup_appointment",
        "lookup_patient_id": "PAT12345",
        "lookup_patient_phone": "03001234567",
        "lookup_patient_name": "Old Patient",
    }
    updates = nodes.understand_intent(pending)

    assert updates["intent"] == "book_appointment"
    assert updates["pending_action"] is None
    assert updates["lookup_patient_id"] is None
    assert updates["lookup_patient_phone"] is None
    assert updates["lookup_patient_name"] is None


def test_checkpoint_stores_history_per_session_and_knowledge_followup_uses_it(monkeypatch):
    from app.graph.graph import _knowledge_module

    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    captured = []

    def fake_answer(question):
        captured.append(question)
        return {"answer": "The ICU hours are 5 PM to 6 PM daily.", "sources": []}

    monkeypatch.setattr(_knowledge_module, "answer_question", fake_answer)
    first = build_graph()
    session_a = {"configurable": {"thread_id": "knowledge-history-a"}}
    session_b = {"configurable": {"thread_id": "knowledge-history-b"}}

    first.invoke({
        "user_message": "Can you repeat the visiting hours I just asked about?",
        "conversation_history": [
            {"role": "user", "content": "What are the ICU visiting hours?"},
            {"role": "assistant", "content": "5 PM to 6 PM daily."},
        ],
    }, config=session_a)
    first.invoke({"user_message": "Does the hospital have a helipad on the roof?"}, config=session_b)
    assert "ICU visiting hours" in captured[0]
    assert "helipad" in captured[1]
    assert first.get_state(session_b).values.get("conversation_history") is None


def test_repeat_those_hours_routes_to_knowledge_with_saved_icu_context(monkeypatch):
    from app.graph.graph import _knowledge_module

    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    retrieval_queries = []

    def fake_answer(question):
        retrieval_queries.append(question)
        return {"answer": "ICU hours: 5:00 PM–6:00 PM daily, one visitor at a time.", "sources": []}

    monkeypatch.setattr(_knowledge_module, "answer_question", fake_answer)
    graph = build_graph()
    config = {"configurable": {"thread_id": "repeat-hours-session"}}
    first_question = "What are the ICU visiting hours?"
    first = graph.invoke({"user_message": first_question}, config=config)
    graph.update_state(config, {"conversation_history": [
        {"role": "user", "content": first_question},
        {"role": "assistant", "content": first["response"]},
    ]})

    followup = "Can you repeat those hours?"
    prior_history = graph.get_state(config).values["conversation_history"]
    result = graph.invoke(
        {"user_message": followup, "conversation_history": prior_history},
        config=config,
    )

    assert result["intent"] == "general_hospital_query"
    assert result["user_message"] == followup
    assert result["response_type"] == "knowledge_answer"
    assert "ICU visiting hours" in retrieval_queries[-1]


def test_repeat_hours_without_prior_context_does_not_use_another_session(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    graph = build_graph()
    hospital_session = {"configurable": {"thread_id": "icu-history-session"}}
    other_session = {"configurable": {"thread_id": "empty-history-session"}}
    graph.invoke({
        "user_message": "What are the ICU visiting hours?",
        "conversation_history": [],
    }, config=hospital_session)
    result = graph.invoke({"user_message": "Can you repeat those hours?"}, config=other_session)

    assert result["intent"] == "unclear_message"
    assert result["response_type"] == "text"


def test_typo_and_grammar_variants_route_to_existing_services(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    cases = [
        ("What time lab open", "general_hospital_query", "knowledge"),
        ("wat docters are avalable?", "list_doctors", "list_doctors"),
        ("emergncy department open at nite?", "general_hospital_query", "knowledge"),
    ]
    for message, expected_intent, expected_route in cases:
        updates = nodes.understand_intent({"user_message": message})
        assert updates["intent"] == expected_intent
        assert route(updates) == expected_route


def test_icu_followups_then_general_ward_timings_use_recent_topic(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    queries = []

    def fake_answer(question):
        queries.append(question)
        answer = (
            "Two people cannot enter the ICU together under the stated limit. Call +92 300 555 0100 to ask the hospital about your request."
            if "main hospital number" in question
            else "The ICU allows a maximum of one visitor at a time."
            if "visitor limit" in question
            else "See the hospital information source."
        )
        return {"answer": answer, "sources": []}

    monkeypatch.setattr(_knowledge_module, "answer_question", fake_answer)
    graph = build_graph()
    config = {"configurable": {"thread_id": "visitor-followups"}}
    sequence = [
        "What are the ICU visiting hours?",
        "how many people can go in?",
        "is there any possibilites to take two persons",
        "What about the general ward?",
        "and what are its timings?",
    ]
    results = [_invoke_with_history(graph, config, message) for message in sequence]

    assert all(result["response_type"] == "knowledge_answer" for result in results)
    assert "one visitor at a time" in results[1]["response"]
    assert "cannot enter the ICU together" in results[2]["response"]
    assert "+92 300 555 0100" in results[2]["response"]
    assert "ICU" in queries[1] and "visitor limit" in queries[1]
    assert "main hospital number" in queries[2]
    assert "general ward" in queries[3].casefold()
    assert "general ward visiting hours" in queries[4].casefold()


@pytest.mark.parametrize(
    "followup",
    [
        "Can I bring two people?",
        "Can more than one person go in together?",
        "Can two people visit together?",
    ],
)
def test_multiple_person_icu_followups_use_icu_policy_only(monkeypatch, followup):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    questions = []
    expected = (
        "No, the ICU allows only one visitor at a time. For further assistance "
        "or to confirm any exceptions, please contact the hospital at "
        "+92 300 555 0100."
    )

    def fake_answer(question):
        questions.append(question)
        assert "ICU" in question
        assert "general ward" not in question.casefold()
        return {"answer": expected, "sources": [{"source": "hospital-info.pdf", "page": 1}]}

    monkeypatch.setattr(_knowledge_module, "answer_question", fake_answer)
    graph = build_graph()
    config = {"configurable": {"thread_id": f"icu-multiple-visitors-{followup}"}}

    first = _invoke_with_history(graph, config, "What are the ICU visiting hours?")
    answer = _invoke_with_history(graph, config, followup)

    assert first["response_type"] == "knowledge_answer"
    assert answer["intent"] == "general_hospital_query"
    assert answer["response_type"] == "knowledge_answer"
    assert answer["response"] == expected
    assert len(questions) == 2
    assert "visitor limit" in questions[-1].casefold()


@pytest.mark.parametrize("service_reply", ["general", "general ward"])
def test_general_ward_answers_service_hours_clarification(monkeypatch, service_reply):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    queries = []

    def fake_answer(question):
        queries.append(question)
        return {
            "answer": "General ward visiting hours are 4:00 PM to 7:00 PM daily, maximum two visitors per patient at a time.",
            "sources": [],
        }

    monkeypatch.setattr(_knowledge_module, "answer_question", fake_answer)
    graph = build_graph()
    config = {"configurable": {"thread_id": "ward-clarification"}}

    first = _invoke_with_history(graph, config, "What are the general ward visiting hours?")
    clarification = _invoke_with_history(graph, config, "Timings/")
    answer = _invoke_with_history(graph, config, service_reply)

    assert first["response_type"] == "knowledge_answer"
    assert "Which department or hospital service" in clarification["response"]
    assert answer["response_type"] == "knowledge_answer"
    assert "4:00 PM to 7:00 PM" in answer["response"]
    assert "maximum two visitors per patient at a time" in answer["response"]
    assert "general ward visiting hours" in queries[-1].casefold()


def test_laboratory_hours_can_be_repeated_but_context_stays_session_scoped(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    queries = []
    monkeypatch.setattr(
        _knowledge_module,
        "answer_question",
        lambda question: (queries.append(question) or {"answer": "Hours are in the guide.", "sources": []}),
    )
    graph = build_graph()
    laboratory = {"configurable": {"thread_id": "lab-session"}}
    isolated = {"configurable": {"thread_id": "unrelated-session"}}

    first = _invoke_with_history(graph, laboratory, "What time lab open")
    repeated = _invoke_with_history(graph, laboratory, "Can you repeat that?")
    isolated_result = _invoke_with_history(graph, isolated, "Can you repeat that?")

    assert first["response_type"] == "knowledge_answer"
    assert repeated["response_type"] == "knowledge_answer"
    assert "lab" in queries[1].casefold()
    assert isolated_result["intent"] == "unclear_message"
    assert isolated_result["response_type"] == "text"


def test_api_session_history_routes_icu_visitor_followup(monkeypatch):
    from uuid import uuid4

    from app import main
    from app.schemas.chat import ChatRequest

    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _knowledge_module,
        "answer_question",
        lambda question: {
            "answer": (
                "The ICU allows a maximum of one visitor at a time."
                if "visitor limit" in question
                else "ICU visiting hours are 5:00 PM to 6:00 PM daily."
            ),
            "sources": [{"source": "hospital-info.pdf", "page": 1, "score": 0.9}],
        },
    )
    session_id = f"icu-followup-{uuid4()}"

    first = main.chat(ChatRequest(
        session_id=session_id,
        message="What are the ICU visiting hours?",
    ))
    followup = main.chat(ChatRequest(
        session_id=session_id,
        message="how many people can go in?",
    ))

    assert first.type == "knowledge_answer"
    assert followup.type == "knowledge_answer"
    assert "one visitor at a time" in followup.response
    saved = main.chat_graph.get_state(
        {"configurable": {"thread_id": session_id}}
    ).values["conversation_history"]
    assert [turn["role"] for turn in saved] == ["user", "assistant", "user", "assistant"]


def test_unknown_service_timings_asks_which_service_without_rag_guess(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    graph = build_graph()
    result = graph.invoke(
        {"user_message": "blorx timings"},
        config={"configurable": {"thread_id": "unknown-service-session"}},
    )

    assert result["intent"] == "unclear_message"
    assert "Which department or hospital service" in result["response"]


def test_facility_question_routes_to_knowledge_even_when_answer_is_unknown(monkeypatch):
    from app.graph.graph import route

    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    update = nodes.understand_intent({
        "user_message": "Does the hospital have a helipad on the roof?"
    })

    assert update["intent"] == "general_hospital_query"
    assert route({**update}) == "knowledge"

