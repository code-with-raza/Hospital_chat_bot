import pytest
from pydantic import ValidationError

from app.tools import patient_tools
from app.graph.nodes import _read_patient_details


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1234567", "1234567"),
        ("+92 (300) 123-4567", "03001234567"),
        ("123456789012345", "123456789012345"),
    ],
)
def test_phone_validation_accepts_supported_boundaries_and_formats(value, expected):
    assert patient_tools.PatientInput(
        name="Test Patient", phone=value, age=30, weight=70
    ).phone == expected


@pytest.mark.parametrize("value", ["123456", "1234567890123456", "abc1234567", "12+34567"])
def test_phone_validation_rejects_invalid_inputs(value):
    with pytest.raises(ValidationError):
        patient_tools.PatientInput(name="Test Patient", phone=value, age=30, weight=70)


@pytest.mark.parametrize("phone", ["1234567", "+92 (300) 123-4567"])
def test_patient_details_parser_accepts_the_same_phone_boundaries(phone):
    fields = _read_patient_details(f"Ali Raza, {phone}, 30 years old, 72 kg")
    assert fields["patient_phone"] == patient_tools.normalize_phone(phone)


def test_save_patient_reuses_existing_record_by_normalized_phone(monkeypatch):
    rows = [{
        "patient_id": "PAT001",
        "name": "Ali Raza",
        "phone": "03001234567",
        "age": 30,
        "weight": 72.0,
    }]
    writes = []
    monkeypatch.setattr(patient_tools, "read_store", lambda *_: rows)
    monkeypatch.setattr(patient_tools, "write_store", lambda *args: writes.append(args))

    same_patient = patient_tools.save_patient(
        "Different Name", "+92 (300) 123-4567", 45, 80
    )

    assert same_patient["patient_id"] == "PAT001"
    assert same_patient["name"] == "Ali Raza"
    assert len(rows) == 1
    assert writes == []

