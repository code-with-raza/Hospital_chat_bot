from typing import TypedDict


class ChatState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_message: str
    intent: str | None
    specialization: str | None
    suggested_specialization: str | None
    doctors: list[dict]
    selected_doctor: dict | None
    appointment_date: str | None
    appointment_time: str | None
    patient_name: str | None
    patient_phone: str | None
    patient_age: int | None
    patient_weight: float | None
    patient_id: str | None
    lookup_patient_name: str | None
    appointment_doctor_filter: str | None
    appointment_specialty_filter: str | None
    response: str
    response_type: str
    response_data: object | None
    sources: list[dict] | None
    pending_action: str | None
    lookup_patient_id: str | None
    lookup_patient_phone: str | None
    lookup_verified_patient_id: str | None
    appointment_time_ambiguous: bool
    appointment_time_hour: str | None
    conversation_history: list[dict]
    resolved_question: str | None
