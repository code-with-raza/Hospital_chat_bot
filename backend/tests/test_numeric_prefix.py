import pytest

from app.graph import nodes
from app.graph.graph import _knowledge_module, build_graph, route


DOCTORS = [
    {"id": "DOC001", "name": "Dr. Hina Malik", "specialization": "Dermatologist"},
    {"id": "DOC002", "name": "Dr. Usman Tariq", "specialization": "Dermatologist"},
]


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("1. Okay, got it, thanks.", "thanks"),
        ("2. What are the laboratory hours?", "general_hospital_query"),
        ("3. What are the ICU visiting hours?", "general_hospital_query"),
        ("1. Can I bring two people to the ICU?", "general_hospital_query"),
    ],
)
def test_leading_list_number_does_not_override_message_intent(
    monkeypatch, message, expected_intent
):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    update = nodes.understand_intent({"user_message": message})

    assert update["intent"] == expected_intent
    assert route(update) == (
        "conversation" if expected_intent == "thanks" else "knowledge"
    )


def test_only_a_standalone_option_selects_a_doctor_while_waiting(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)

    selection = nodes.understand_intent({
        "user_message": "2",
        "pending_action": "doctor",
        "doctors": DOCTORS,
    })
    no_pending_selection = nodes.understand_intent({
        "user_message": "2",
        "doctors": DOCTORS,
    })

    assert selection["selected_doctor"] == DOCTORS[1]
    assert selection["pending_action"] == "date"
    assert no_pending_selection["intent"] == "unclear_message"
    assert "selected_doctor" not in no_pending_selection


def test_numbers_inside_questions_are_not_doctor_selections(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    update = nodes.understand_intent({
        "user_message": "What are the visiting hours for 2 people?",
        "pending_action": "doctor",
        "doctors": DOCTORS,
    })

    assert update["intent"] == "general_hospital_query"
    assert "selected_doctor" not in update
    assert update.get("pending_action") != "date"


def test_numbered_icu_question_keeps_raw_message_and_pending_booking(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    requested = []
    monkeypatch.setattr(
        _knowledge_module,
        "answer_question",
        lambda question: (
            requested.append(question)
            or {"answer": "The ICU allows one visitor at a time.", "sources": []}
        ),
    )
    graph = build_graph()
    config = {"configurable": {"thread_id": "numbered-icu-pending-booking"}}
    original = "3. What are the ICU visiting hours?"
    result = graph.invoke({
        "user_message": original,
        "conversation_history": [],
        "pending_action": "date",
        "appointment_date": "2027-01-12",
        "selected_doctor": DOCTORS[0],
    }, config=config)
    saved = graph.get_state(config).values

    assert result["user_message"] == original
    assert result["response_type"] == "knowledge_answer"
    assert "ICU" in requested[0]
    assert saved["pending_action"] == "date"
    assert saved["appointment_date"] == "2027-01-12"


def test_only_numbered_list_markers_at_the_start_are_removed():
    assert nodes._strip_leading_list_number("1. Okay, thanks.") == "Okay, thanks."
    assert nodes._strip_leading_list_number("2) What time is the lab open?") == "What time is the lab open?"
    assert nodes._strip_leading_list_number("3 - What are ICU hours?") == "What are ICU hours?"
    assert nodes._strip_leading_list_number("What are the visiting hours for 2 people?") == "What are the visiting hours for 2 people?"
