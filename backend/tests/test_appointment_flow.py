from datetime import date, timedelta

from app.graph import nodes
from app.tools.appointment_tools import normalize_time_input


DOCTOR = {
    "id": "DOC001",
    "name": "Dr. Test Doctor",
    "specialization": "Cardiologist",
    "working_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "appointment_slots": ["10:00", "10:30", "13:00", "22:00"],
}


def test_new_booking_keeps_date_and_time_from_the_current_message(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    updates = nodes.understand_intent({
        "user_message": "Book an appointment with the same doctor coming Tuesday at 10 AM",
        "selected_doctor": DOCTOR,
        "specialization": "Cardiologist",
        "patient_id": "PAT001",
        "patient_name": "Ali Raza",
        "patient_phone": "03001234567",
        "patient_age": 30,
        "patient_weight": 72,
    })

    assert updates["intent"] == "book_appointment"
    assert updates["selected_doctor"] == DOCTOR
    assert updates["specialization"] == "Cardiologist"
    assert updates["appointment_date"] is not None
    assert updates["appointment_time"].casefold() == "10 am"


def test_new_booking_clears_old_date_and_time_when_message_has_none(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    updates = nodes.understand_intent({
        "user_message": "Book an appointment with the same doctor",
        "selected_doctor": DOCTOR,
        "specialization": "Cardiologist",
        "appointment_date": "2020-01-01",
        "appointment_time": "10:00",
    })

    assert updates["appointment_date"] is None
    assert updates["appointment_time"] is None
    assert updates["selected_doctor"] == DOCTOR


def test_weekday_followup_continues_pending_booking_but_hospital_question_does_not(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    base_state = {"pending_action": "date", "selected_doctor": DOCTOR}

    date_turn = nodes.understand_intent({**base_state, "user_message": "Tuesday"})
    unrelated_turn = nodes.understand_intent({
        **base_state,
        "user_message": "What are the visiting hours?",
    })

    assert date_turn["intent"] == "continue_booking"
    assert date_turn["appointment_date"]
    assert unrelated_turn["intent"] == "general_hospital_query"


def test_ambiguous_bare_hour_waits_for_meridiem_before_checking_availability(monkeypatch):
    future = date.today() + timedelta(days=1)
    monkeypatch.setattr(nodes, "check_availability", lambda *args: (_ for _ in ()).throw(AssertionError("checked too early")))
    result = nodes.continue_booking({
        "selected_doctor": DOCTOR,
        "appointment_date": future.isoformat(),
        "appointment_time": None,
        "appointment_time_ambiguous": True,
        "appointment_time_hour": "10:00",
        "user_message": "10",
    })
    assert "10:00 AM or 10:00 PM" in result["response"]
    assert result["pending_action"] == "time"


def test_meridiem_reply_resolves_ambiguous_time_from_pending_context(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    updates = nodes.understand_intent({
        "user_message": "PM",
        "pending_action": "time",
        "appointment_time_ambiguous": True,
        "appointment_time_hour": "10:00",
    })
    assert updates["intent"] == "continue_booking"
    assert updates["appointment_time"] == "10:00 PM"
    assert updates["appointment_time_ambiguous"] is False


def test_explicit_am_pm_times_keep_their_meaning_and_bare_hour_is_rejected():
    slots = ["10:00", "22:00", "13:00"]
    assert normalize_time_input("10 AM", slots) == "10:00"
    assert normalize_time_input("10 PM", slots) == "22:00"
    assert normalize_time_input("13:00", slots) == "13:00"
    assert normalize_time_input("10", slots) is None


def test_existing_phone_patient_is_resolved_before_booking_state_is_saved(monkeypatch):
    existing = {
        "patient_id": "PAT009", "name": "Ali Raza", "phone": "03001234567",
        "age": 30, "weight": 72.0,
    }
    captured = {}
    monkeypatch.setattr(nodes, "check_availability", lambda *args: {"available": True})
    monkeypatch.setattr(nodes, "find_patient_by_phone", lambda _: existing)
    monkeypatch.setattr(nodes, "get_patient", lambda *_args, **_kwargs: None)

    def save(state, doctor, appointment_date, appointment_time):
        captured.update(state)
        return {"response": "saved"}

    monkeypatch.setattr(nodes, "_save_booking", save)
    future = date.today() + timedelta(days=1)
    while future.strftime("%A") not in DOCTOR["working_days"]:
        future += timedelta(days=1)
    nodes.continue_booking({
        "selected_doctor": DOCTOR,
        "appointment_date": future.isoformat(),
        "appointment_time": "10 AM",
        "patient_id": None,
        "patient_name": "Ali Raza",
        "patient_phone": "+92 (300) 123-4567",
        "patient_age": 30,
        "patient_weight": 72,
    })
    assert captured["patient_id"] == "PAT009"


def test_patient_details_followup_preserves_slot_and_saves_booking(monkeypatch):
    future = date(2026, 10, 19)
    captured = {}
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nodes, "check_availability", lambda *args: {"available": True})
    monkeypatch.setattr(nodes, "find_patient_by_phone", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nodes, "get_patient", lambda *_args, **_kwargs: None)

    def save(state, doctor, appointment_date, appointment_time):
        captured.update(date=appointment_date, time=appointment_time, **state)
        return {"response": "booked", "response_type": "appointment_confirmation"}

    monkeypatch.setattr(nodes, "_save_booking", save)
    prior = {
        "pending_action": "patient_info",
        "selected_doctor": DOCTOR,
        "appointment_date": future.isoformat(),
        "appointment_time": "10:00",
    }
    update = nodes.understand_intent({
        **prior,
        "user_message": "Usman 03000000000, 32 years old, 68 kg",
    })
    merged = {**prior, **update}
    result = nodes.continue_booking(merged)

    assert result["response_type"] == "appointment_confirmation"
    assert captured["date"] == future.isoformat()
    assert captured["time"] == "10:00"
    assert captured["patient_name"] == "Usman"
    assert captured["patient_phone"] == "03000000000"
    assert captured["patient_age"] == 32
    assert captured["patient_weight"] == 68.0


def test_patient_details_parser_does_not_leave_age_or_weight_labels_in_name():
    details = nodes._read_patient_details("Usman 03000000000, 32 years old, 68 kg")
    assert details == {
        "patient_name": "Usman",
        "patient_phone": "03000000000",
        "patient_age": 32,
        "patient_weight": 68.0,
    }


def test_rebooking_prompt_does_not_reset_current_booking_step(monkeypatch):
    monkeypatch.setattr(nodes, "understand_with_llm", lambda *_args, **_kwargs: None)
    future = date(2026, 10, 19).isoformat()
    state = {
        "user_message": "I want to book an appointment with this doctor",
        "pending_action": "date",
        "selected_doctor": DOCTOR,
        "appointment_date": future,
        "appointment_time": "10:00",
        "specialization": "Cardiologist",
    }
    updates = nodes.understand_intent(state)

    assert updates["intent"] == "continue_booking"
    assert updates.get("appointment_date", state["appointment_date"]) == future
    assert updates.get("appointment_time", state["appointment_time"]) == "10:00"
    assert updates.get("selected_doctor", state["selected_doctor"]) == DOCTOR
