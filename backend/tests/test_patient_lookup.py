from app.graph import nodes


PATIENT = {
    "patient_id": "PAT001", "name": "Ali Raza", "phone": "03001234567",
    "age": 30, "weight": 72.0,
}


def test_patient_id_alone_does_not_look_up_or_reveal_record(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes, "get_patient", lambda patient_id: calls.append(patient_id))
    result = nodes.get_appointment_node({"lookup_patient_id": "PAT001"})
    assert calls == []
    assert "registered phone" in result["response"]
    assert "PAT001" not in result["response"]


def test_name_alone_cannot_access_patient_details(monkeypatch):
    monkeypatch.setattr(nodes, "get_patient", lambda _: PATIENT)
    result = nodes.get_patient_details_node({"lookup_patient_name": "Ali Raza"})
    assert result.get("response_data") is None
    assert "both your patient ID" in result["response"]
    assert "Ali Raza" not in result["response"]


def test_mismatched_id_and_phone_return_generic_failure(monkeypatch):
    monkeypatch.setattr(nodes, "get_patient", lambda _: PATIENT)
    monkeypatch.setattr(nodes, "find_patient_by_phone", lambda _: None)
    result = nodes.get_appointment_node({
        "lookup_patient_id": "PAT001",
        "lookup_patient_phone": "03111111111",
    })
    assert "couldn't verify" in result["response"]
    assert "PAT001" not in result["response"]
    assert "Ali Raza" not in result["response"]
    assert result["lookup_patient_id"] is None


def test_matching_id_and_phone_reveal_appointments_then_clear_verification_state(monkeypatch):
    appointment = {
        "appointment_id": "APT001", "patient_id": "PAT001", "doctor_id": "DOC001",
        "doctor_name": "Dr. Test", "specialization": "Cardiologist",
        "date": "2030-01-01", "time": "10:00", "status": "confirmed",
    }
    monkeypatch.setattr(nodes, "get_patient", lambda _: PATIENT)
    monkeypatch.setattr(nodes, "find_patient_by_phone", lambda _: PATIENT)
    monkeypatch.setattr(nodes, "get_patient_appointments", lambda _: [appointment])
    result = nodes.get_appointment_node({
        "lookup_patient_id": "PAT001",
        "lookup_patient_phone": "03001234567",
    })
    assert result["response_data"] == [appointment]
    assert result["lookup_patient_id"] is None
    assert result["lookup_patient_phone"] is None
    assert result["lookup_verified_patient_id"] is None


def test_patient_id_and_phone_are_collected_across_followup_turns(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "understand_with_llm",
        lambda _: (_ for _ in ()).throw(AssertionError("Patient credentials must stay local.")),
    )
    first = nodes.understand_intent({"user_message": "PAT001"})
    assert first["lookup_patient_id"] == "PAT001"
    assert first.get("patient_name") is None

    second = nodes.understand_intent({
        "user_message": "03001234567",
        "pending_action": "lookup_appointment",
        "lookup_patient_id": "PAT001",
    })
    assert second["lookup_patient_id"] == "PAT001"
    assert second["lookup_patient_phone"] == "03001234567"
    assert second["intent"] == "check_appointment"
