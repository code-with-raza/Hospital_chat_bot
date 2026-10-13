import re

from app.tools.storage import read_store

ALIASES = {
    "heart doctor": "Cardiologist",
    "heart specialist": "Cardiologist",
    "cardiology": "Cardiologist",
    "skin doctor": "Dermatologist",
    "skin specialist": "Dermatologist",
    "dermatology": "Dermatologist",
    "brain doctor": "Neurologist",
    "brain specialist": "Neurologist",
    "neurology": "Neurologist",
    "bone doctor": "Orthopedic Specialist",
    "orthopadic": "Orthopedic Specialist",
    "orthopedic doctor": "Orthopedic Specialist",
    "orthopedist": "Orthopedic Specialist",
    "orthopedic": "Orthopedic Specialist",
    "ent": "ENT Specialist",
    "ear nose throat": "ENT Specialist",
    "children doctor": "Pediatrician",
    "child specialist": "Pediatrician",
    "children specialist": "Pediatrician",
    "child": "Pediatrician",
    "kids specialist": "Pediatrician",
    "pediatric doctor": "Pediatrician",
    "child doctor": "Pediatrician",
    "kids doctor": "Pediatrician",
    "general doctor": "General Physician",
    "family doctor": "General Physician",
    "physiotherapy": "Physiotherapist",
    "physical therapy": "Physiotherapist",
    "physiotherapist": "Physiotherapist",
    "gynecologist": "Gynecologist",
    "gynaecologist": "Gynecologist",
}


def normalize_specialization(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    for phrase, specialization in ALIASES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return specialization
    doctors = read_store("doctors.json", "doctors")
    for doctor in doctors:
        specialization = doctor["specialization"]
        if specialization.lower() in text:
            return specialization
    return value.strip().title()


def find_doctors(specialization: str) -> list[dict]:
    normalized = normalize_specialization(specialization) or specialization
    doctors = read_store("doctors.json", "doctors")
    return [
        doctor
        for doctor in doctors
        if doctor.get("specialization", "").casefold() == normalized.casefold()
    ]


def list_doctors() -> list[dict]:
    """Return the full hospital directory for general availability questions."""
    return read_store("doctors.json", "doctors")


def get_doctor(doctor_id: str) -> dict | None:
    doctors = read_store("doctors.json", "doctors")
    return next((doctor for doctor in doctors if doctor.get("id") == doctor_id), None)
