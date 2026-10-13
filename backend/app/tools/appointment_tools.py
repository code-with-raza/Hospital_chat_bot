import re
from datetime import date as date_type
from datetime import datetime
from threading import Lock

from app.tools.doctor_tools import get_doctor
from app.tools.storage import next_id, read_store, write_store

_booking_lock = Lock()


def format_time(time: str) -> str:
    """Display a stored 24-hour time in a familiar 12-hour format."""
    try:
        return datetime.strptime(time, "%H:%M").strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return time


def get_available_slots(doctor_id: str, date: str) -> list[str]:
    doctor = get_doctor(doctor_id)
    if not doctor:
        return []
    return [
        time
        for time in doctor.get("appointment_slots", [])
        if check_availability(doctor_id, date, time)["available"]
    ]


def normalize_time_input(value: str, available_slots: list[str]) -> str | None:
    """Convert a patient's AM/PM or 24-hour entry to the stored HH:MM format."""
    match = re.search(
        r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    meridiem = (match.group(3) or "").replace(".", "").lower()
    if minute > 59:
        return None

    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12
        if meridiem.startswith("p"):
            hour += 12
    elif not 0 <= hour <= 23:
        return None
    elif 1 <= hour <= 12:
        # Without an AM/PM marker, 1–12 cannot be resolved safely.
        return None

    requested = f"{hour:02d}:{minute:02d}"
    if requested in available_slots:
        return requested

    return requested


def ambiguous_hour_from_text(value: str) -> str | None:
    """Return an unqualified 1–12 hour token, retaining optional minutes."""
    match = re.search(
        r"(?:\bat\s+|@\s*|^\s*)(1[0-2]|0?[1-9])(?::([0-5]\d))?"
        r"(?!\s*(?:a\.?m\.?|p\.?m\.?))(?:\b|$)",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    hour = int(match.group(1))
    minute = match.group(2) or "00"
    return f"{hour}:{minute}"

def check_availability(doctor_id: str, date: str, time: str) -> dict:
    doctor = get_doctor(doctor_id)
    if not doctor:
        return {"available": False, "message": "I couldn't find that doctor."}
    try:
        appointment_date = date_type.fromisoformat(date)
    except (TypeError, ValueError):
        return {"available": False, "message": "Please provide a date in YYYY-MM-DD format."}
    if appointment_date < date_type.today():
        return {"available": False, "message": "That date has already passed. Please choose a future date."}
    day = appointment_date.strftime("%A")
    working_days = doctor.get("working_days", [])
    if day not in working_days:
        schedule = ", ".join(working_days)
        return {
            "available": False,
            "message": f"{doctor['name']} does not work on {day}. Working days: {schedule}.",
        }
    if time not in doctor.get("appointment_slots", []):
        slots = ", ".join(doctor["appointment_slots"])
        return {
            "available": False,
            "message": f"That time is not a listed appointment slot. Available times: {slots}.",
        }
    appointments = read_store("appointments.json", "appointments")
    booked = any(
        appointment.get("doctor_id") == doctor_id
        and appointment.get("date") == date
        and appointment.get("time") == time
        and appointment.get("status") == "confirmed"
        for appointment in appointments
    )
    if booked:
        return {
            "available": False,
            "message": "This appointment slot is already booked. Please choose another time.",
        }
    return {"available": True, "message": "Appointment slot is available."}


def book_appointment(patient_id: str, doctor_id: str, date: str, time: str) -> dict:
    with _booking_lock:
        availability = check_availability(doctor_id, date, time)
        if not availability["available"]:
            raise ValueError(availability["message"])

        doctor = get_doctor(doctor_id)
        appointments = read_store("appointments.json", "appointments")
        appointment = {
            "appointment_id": next_id("APT", appointments, "appointment_id"),
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "doctor_name": doctor["name"],
            "specialization": doctor["specialization"],
            "date": date,
            "time": time,
            "status": "confirmed",
        }
        appointments.append(appointment)
        write_store("appointments.json", "appointments", appointments)
        return appointment


def get_patient_appointments(patient_id: str) -> list[dict]:
    appointments = read_store("appointments.json", "appointments")
    return [
        appointment
        for appointment in appointments
        if appointment.get("patient_id", "").casefold() == patient_id.casefold()
    ]
