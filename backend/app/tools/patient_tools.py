import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.tools.storage import next_id, read_store, write_store


def normalize_phone(phone: str) -> str:
    """Return the application's canonical phone form (7–15 digits)."""
    candidate = (phone or "").strip()
    if not candidate or not re.fullmatch(r"\+?[\d\s()-]+", candidate):
        raise ValueError("Enter a valid phone number (7 to 15 digits).")
    digits = re.sub(r"\D", "", candidate)
    # The demo data uses Pakistan's domestic mobile form (03xx...). Store
    # +92 numbers in that same form so equivalent spellings match.
    if candidate.startswith("+") and digits.startswith("92") and len(digits) == 12:
        digits = "0" + digits[2:]
    if not 7 <= len(digits) <= 15:
        raise ValueError("Enter a valid phone number (7 to 15 digits).")
    return digits


def extract_phone_number(text: str) -> str | None:
    """Extract a phone number using the same digit-count rules as validation."""
    for match in re.finditer(r"(?<!\w)\+?\d[\d ()-]*\d(?!\w)|(?<!\w)\d{7}(?!\w)", text):
        try:
            return normalize_phone(match.group(0))
        except ValueError:
            continue
    return None


class PatientInput(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    phone: str
    age: int = Field(ge=0, le=120)
    weight: float = Field(gt=0, le=500)

    @field_validator("phone")
    @classmethod
    def valid_phone(cls, value: str) -> str:
        return normalize_phone(value)


def save_patient(name: str, phone: str, age: int, weight: float) -> dict:
    patient = PatientInput(name=name, phone=phone, age=age, weight=weight).model_dump()
    patients = read_store("patients.json", "patients")
    existing = next(
        (
            row for row in patients
            if _stored_phone_matches(row.get("phone", ""), patient["phone"])
        ),
        None,
    )
    if existing:
        return existing
    patient["patient_id"] = next_id("PAT", patients, "patient_id")
    patients.append(patient)
    write_store("patients.json", "patients", patients)
    return patient


def get_patient(patient_id: str) -> dict | None:
    patients = read_store("patients.json", "patients")
    return next(
        (
            patient
            for patient in patients
            if patient.get("patient_id", "").casefold() == patient_id.casefold()
        ),
        None,
    )


def find_patient_by_phone(phone: str) -> dict | None:
    digits = normalize_phone(phone)
    patients = read_store("patients.json", "patients")
    return next(
        (
            patient
            for patient in patients
            if _stored_phone_matches(patient.get("phone", ""), digits)
        ),
        None,
    )


def _stored_phone_matches(stored_phone: str, requested_digits: str) -> bool:
    try:
        return normalize_phone(stored_phone) == requested_digits
    except ValueError:
        return False
