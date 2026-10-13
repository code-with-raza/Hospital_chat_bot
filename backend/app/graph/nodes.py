import calendar
import json
import re
from difflib import SequenceMatcher
from datetime import date, timedelta
from typing import Any

from pydantic import ValidationError

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.tools.appointment_tools import (
    book_appointment,
    ambiguous_hour_from_text,
    check_availability,
    format_time,
    get_available_slots,
    get_patient_appointments,
    normalize_time_input,
)
from app.tools.doctor_tools import ALIASES, find_doctors, list_doctors, normalize_specialization
from app.tools.patient_tools import (
    extract_phone_number,
    find_patient_by_phone,
    get_patient,
    normalize_phone,
    save_patient,
)


def _safe_classification_history(
    history: list[dict[str, Any]] | None,
) -> list[str]:
    """Return recent public-service context, excluding patient/booking details."""
    if not history:
        return []

    sensitive = re.compile(
        r"\b(?:patient|appointment|book(?:ing)?|phone|weight|age|PAT\d+|"
        r"\d{4}-\d{2}-\d{2})\b|(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)",
        re.I,
    )
    public_topic = re.compile(
        r"\b(?:hospital|doctor|specialist|department|service|hours?|timings?|"
        r"visiting|visitor|ward|icu|laboratory|lab|pharmacy|emergency|policy)\b",
        re.I,
    )
    relevant = []
    for turn in history[-8:]:
        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content or sensitive.search(content):
            continue
        if public_topic.search(content):
            relevant.append(f"{role.title()}: {content[:500]}")
    return relevant[-6:]


def understand_with_llm(
    message: str, history: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Classify a message and extract relevant details using the LLM."""
    if not LLM_API_KEY:
        return None

    instructions = (
        "Classify the user's message for a hospital assistant. "
        "Return a JSON object with intent, resolved_question, specialization, date in "
        "YYYY-MM-DD, time in HH:MM, patient_name, patient_phone, "
        "patient_age, and patient_weight. Use null for unknown values. "
        "Valid intents are: greeting, general_chat, unclear_message, "
        "help, thanks, symptom_guidance, list_doctors, find_doctor, "
        "book_appointment, check_appointment, check_patient_details, "
        "privacy_boundary, cancel_appointment, general_hospital_query. "
        "Use greeting for greetings, general_chat for casual conversation "
        "or emotional sharing, and unclear_message when the meaning is "
        "unclear. Use help when the user asks what you can do and thanks "
        "when the user expresses gratitude. Use symptom_guidance when "
        "the user describes a symptom, injury, pain, or health concern, "
        "even without asking for a doctor. Use find_doctor when the user "
        "asks to find or see a doctor. Use book_appointment to schedule "
        "a new appointment and check_appointment for an existing one. "
        "Use check_patient_details for requests about the user's own "
        "patient record, cancel_appointment for cancellation requests, "
        "and general_hospital_query for hospital policies, visiting "
        "hours, departments, facilities, and services. Use "
        "privacy_boundary when the user requests other patients' "
        "private information. For knee pain, an orthopedic specialist "
        "may be appropriate. Do not diagnose conditions or claim "
        "certainty about symptoms. Never invent hospital policies, "
        "doctors, or appointment information. Extract only details "
        "explicitly provided in the current user message. Recent conversation "
        "is context only: use it to understand a pronoun or follow-up, but do "
        "not carry an old booking or patient detail into this turn. For a "
        "contextual hospital question, resolved_question may contain a concise "
        "standalone retrieval query. Do not answer it or add facts. Set it to "
        "null if the topic is genuinely unclear; otherwise it may be null. "
        "Classify only the current user message."
    )

    try:
        from openai import OpenAI

        client_options = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            client_options["base_url"] = LLM_BASE_URL

        client = OpenAI(**client_options)
        context = _safe_classification_history(history)
        current_message = message
        if context:
            current_message = (
                "Recent conversation for reference only:\n"
                + "\n".join(context)
                + "\n\nCurrent user message to classify:\n"
                + message
            )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": current_message},
            ],
        )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        if not isinstance(data, dict):
            return None

        valid_intents = {
            "greeting", "general_chat", "unclear_message", "help", "thanks",
            "symptom_guidance", "list_doctors", "find_doctor", "book_appointment",
            "check_appointment", "check_patient_details", "privacy_boundary",
            "cancel_appointment", "general_hospital_query",
        }
        if data.get("intent") not in valid_intents:
            return None
        resolved_question = data.get("resolved_question")
        data["resolved_question"] = (
            resolved_question.strip()[:500]
            if isinstance(resolved_question, str) and resolved_question.strip()
            else None
        )

        return data

    except (ValidationError, json.JSONDecodeError, ValueError):
        return None
    except Exception:
        return None

def understand_locally(message: str) -> dict[str, Any]:
    text = message.strip()
    lower_text = text.lower()

    emergency_patterns = (
        r"\bchest pain\b",
        r"\b(?:difficulty|trouble) breathing\b",
        r"\bshortness of breath\b",
        r"\bcan't breathe\b",
        r"\bcannot breathe\b",
    )

    if any(re.search(pattern, lower_text) for pattern in emergency_patterns):
        intent = "symptom_guidance"

    elif re.fullmatch(r"PAT\d+", text.strip(".! "), re.I):
        intent = "check_appointment"

    elif re.fullmatch(
        r"(?:hi|hello|hey|hi there|hello there|hey there|"
        r"good morning|good afternoon|good evening)[!. ]*",
        lower_text,
    ):
        intent = "general_chat"

    elif re.fullmatch(
        r"(?:thanks|thank you|thanks a lot|thank you so much|"
        r"many thanks|thank you for helping me)[!. ]*",
        lower_text,
    ):
        intent = "thanks"

    elif re.fullmatch(
        r"(?:(?:okay|ok|alright|got it|understood)[, ]+)+(?:thanks|thank you)[!. ]*",
        lower_text,
    ):
        intent = "thanks"

    elif re.fullmatch(r"(?:okay|ok|alright|got it|understood|sure|sounds good)[!. ]*", lower_text):
        intent = "general_chat"

    elif any(
        phrase in lower_text
        for phrase in (
            "how can you help",
            "what can you help me with",
            "what can you do",
            "what services do you provide",
            "what can i ask",
        )
    ):
        intent = "help"

    elif any(
        phrase in lower_text
        for phrase in (
            "can you diagnose",
            "diagnose my symptoms",
            "give me a diagnosis",
            "what is my diagnosis",
        )
    ):
        intent = "symptom_guidance"

    elif any(
        phrase in lower_text
        for phrase in (
            "list all doctors",
            "list the doctors",
            "show all doctors",
            "show me all doctors",
            "which doctors are available",
            "what doctors are available",
            "available doctors",
            "list doctors",
        )
    ):
        intent = "list_doctors"

    elif (
        _has_fuzzy_word(lower_text, {"what", "which"})
        and _has_fuzzy_word(lower_text, {"doctor", "doctors"})
        and _has_fuzzy_word(lower_text, {"available"})
    ):
        intent = "list_doctors"

    elif (
        any(
            phrase in lower_text
            for phrase in (
                "every patient",
                "all patients",
                "other patients",
                "all patient records",
                "everyone's phone",
                "every patient's phone",
            )
        )
        and any(
            word in lower_text
            for word in (
                "phone",
                "appointment",
                "details",
                "records",
                "personal information",
                "patient information",
            )
        )
    ):
        intent = "privacy_boundary"

    elif "cancel" in lower_text or "cancellation" in lower_text:
        intent = "cancel_appointment"

    elif any(
        phrase in lower_text
        for phrase in (
            "my patient details",
            "my patient information",
            "show my details",
            "show me my details",
            "my personal information",
            "my patient record",
            "my profile",
        )
    ):
        intent = "check_patient_details"

    elif any(
        phrase in lower_text
        for phrase in (
            "book an appointment",
            "book appointment",
            "schedule an appointment",
            "make an appointment",
            "i want an appointment",
            "i want to book",
            "i need to book",
            "appointment instead",
        )
    ):
        intent = "book_appointment"

    elif (
        any(
            phrase in lower_text
            for phrase in (
                "check my appointment",
                "check appointment",
                "my appointment status",
                "show my appointment",
                "show my appointments",
                "my appointments",
                "appointment details",
                "when is my appointment",
                "when is the appointment",
                "find my appointment",
                "look up my appointment",
            )
        )
        or (
            "appointment" in lower_text
            and re.search(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)", text)
        )
    ):
        intent = "check_appointment"

    elif any(
        phrase in lower_text
        for phrase in (
            "my day has been",
            "i feel stressed",
            "i am stressed",
            "i'm stressed",
            "i feel sad",
            "i am sad",
            "i'm sad",
            "i am worried",
            "i'm worried",
            "i feel worried",
            "how are you",
            "what's up",
            "what is up",
        )
    ):
        intent = "general_chat"

    elif any(
        term in lower_text
        for term in (
            "visiting hour",
            "visiting time",
            "visitor",
            "visitors",
            "icu",
            "intensive care",
            "head nurse",
            "nurse in charge",
            "nurse responsible",
            "department hours",
            "department timing",
            "department timings",
            "operating hours",
            "opening hours",
            "working hours",
            "what are the hours",
            "what are the timings",
            "what are the timing",
            "what time does",
            "what time is",
            "when does the",
            "when is the",
            "laboratory",
            "lab hours",
            "radiology",
            "imaging",
            "emergency department",
            "emergency room",
            "hospital policy",
            "hospital information",
            "hospital facilities",
            "hospital services",
            "hospital rules",
            "transplant success rate",
            "kidney transplant",
            "hospital address",
            "hospital location",
            "parking",
            "cafeteria",
            "admission process",
            "discharge process",
        )
    ):
        intent = "general_hospital_query"

    elif (
        _has_fuzzy_word(lower_text, {"lab", "laboratory"})
        and _has_fuzzy_word(lower_text, {"hour", "hours", "open", "opening", "time", "timing", "timings"})
    ) or (
        _has_fuzzy_word(lower_text, {"emergency"})
        and _has_fuzzy_word(lower_text, {"department"})
        and _has_fuzzy_word(lower_text, {"hour", "hours", "open", "night"}, aliases={"night": {"nite"}})
    ):
        intent = "general_hospital_query"

    elif (
        re.search(r"\b(?:hospital|helipad|rooftop|facility|facilities|parking|cafeteria)\b", lower_text)
        and re.search(r"\b(?:does|do|is|are|has|have|can|what|where|when|how)\b", lower_text)
    ):
        # Route plausible facility questions to RAG so missing facts receive
        # the knowledge node's explicit unavailable response.
        intent = "general_hospital_query"

    elif re.search(
        r"\b(?:pain|ache|aching|fever|cough|cold|headache|migraine|"
        r"rash|itching|itchy|swelling|dizziness|dizzy|nausea|"
        r"vomiting|diarrhea|diarrhoea|fatigue|stomach pain|"
        r"back pain|knee pain|neck pain|skin problem|skin irritation)\b",
        lower_text,
    ):
        intent = "symptom_guidance"

    elif (
        re.search(
            r"\b(?:doctor|doctors|specialist|specialists|physician)\b",
            lower_text,
        )
        or re.search(
            r"\b(?:cardio|dermato|neuro|ortho|pediatric|paediatric|"
            r"gastro|gynec|gynaec|radiolog|oncolog|urolog|"
            r"ophthalmolog|dentist|psychiatr|psycholog|pulmonolog|"
            r"nephrolog|endocrinolog|ent)\w*\b",
            lower_text,
        )
        or re.search(r"\b\w+ologist\b", lower_text)
        or any(
            phrase in lower_text
            for phrase in (
                "find me a doctor",
                "find a doctor",
                "need a doctor",
                "want a doctor",
                "looking for a doctor",
                "see a doctor",
                "which doctor",
                "what doctor",
                "doctor for",
            )
        )
    ):
        intent = "find_doctor"

    else:
        intent = "unclear_message"

    specialization = _requested_specialization(text)
    suggested_specialization = _suggest_specialization(
        text,
        specialization,
    )

    if suggested_specialization and intent == "find_doctor":
        intent = "confirm_specialization"

    weekday = _find_weekday(lower_text)

    result: dict[str, Any] = {
        "intent": intent,
        "specialization": specialization,
        "suggested_specialization": suggested_specialization,
        "appointment_date": None,
        "appointment_time": None,
        "patient_phone": None,
        "patient_age": None,
        "patient_weight": None,
        "patient_name": None,
        "patient_id": None,
        "lookup_patient_id": None,
        "lookup_patient_name": None,
        "appointment_doctor_filter": None,
        "appointment_specialty_filter": None,
    }

    if weekday:
        result["weekday"] = weekday

    date_match = re.search(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})\b",
        text,
    )
    if date_match:
        result["appointment_date"] = date_match.group(0)

    time_match = re.search(
        r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b",
        text,
        re.I,
    )
    if time_match:
        result["appointment_time"] = time_match.group(0).strip()

    phone = extract_phone_number(text)
    if phone:
        result["patient_phone"] = phone

    age_match = re.search(
        r"\b(?:age|i am|i'm|aged)\s*[:=]?\s*(\d{1,3})\b",
        lower_text,
    )
    if age_match:
        result["patient_age"] = int(age_match.group(1))

    weight_match = re.search(
        r"\b(?:weight|weigh)\s*[:=]?\s*"
        r"(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs|kilograms)?\b",
        lower_text,
    )
    if weight_match:
        result["patient_weight"] = float(weight_match.group(1))

    patient_id_match = re.search(r"\bPAT\d+\b", text, re.I)
    if patient_id_match:
        patient_id = patient_id_match.group(0).upper()
        result["patient_id"] = patient_id
        result["lookup_patient_id"] = patient_id

    return result


def _has_fuzzy_word(
    text: str,
    choices: set[str],
    *,
    aliases: dict[str, set[str]] | None = None,
) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    aliases = aliases or {}
    for word in words:
        for choice in choices:
            if word in aliases.get(choice, set()) or SequenceMatcher(None, word, choice).ratio() >= 0.8:
                return True
    return False


def _strip_leading_list_number(message: str) -> str:
    """Strip a leading numbered-list marker, leaving other numbers untouched."""
    return re.sub(r"^\s*\d+\s*(?:[.)]\s+|\s+-\s+)", "", message, count=1).strip()


def _is_contextual_hospital_followup(message: str, history: list[dict[str, Any]]) -> bool:
    if not history:
        return False
    recent_topic = _recent_hospital_topic(history)
    if not recent_topic:
        return False
    asks_about_multiple_visitors = (
        re.search(r"\b(?:more than one|two|2|multiple|several)\b", message)
        and re.search(r"\b(?:people|persons?|visitors?)\b", message)
        and re.search(r"\b(?:bring|go|enter|visit|together|inside|in|allow)\b", message)
    )
    return bool(
        re.search(r"\brepeat\b.*\b(?:that|those|it|hours?|times?)\b", message)
        or re.search(r"\bwhat about\b.*\b(?:icu|ward|laboratory|lab|department)\b", message)
        or (
            re.search(r"\bhow many\b.*\b(?:people|visitors?)\b|\bpeople can go in\b", message)
            and re.search(r"\b(?:icu|visiting|visitor|ward)\b", recent_topic)
        )
        or (
            re.search(
                r"\b(?:possibilit\w*|possible|can|could)\b.*\b(?:two|2)\s+(?:people|persons?|visitors?)\b",
                message,
            )
            and re.search(r"\b(?:icu|intensive care)\b", recent_topic)
        )
        or (
            asks_about_multiple_visitors
            and re.search(r"\b(?:icu|intensive care)\b", recent_topic)
        )
        or re.search(r"\b(?:its|their)\s+(?:hours?|timings?|times?)\b|\band\s+what\s+are\s+its\s+(?:hours?|timings?|times?)\b", message)
        or re.search(r"\bcan you repeat that\b", message)
    )


def _recent_hospital_topic(history: list[dict[str, Any]]) -> str:
    topic = re.compile(
        r"\b(?:hospital|icu|intensive care|visiting|visitor|ward|policy|hours?|timings?|"
        r"facility|facilities|department|pharmacy|emergency|lab|laboratory)\b",
        re.I,
    )
    followup = re.compile(
        r"\b(?:repeat|what about|how many|people can go in|possibilit\w*|"
        r"two\s+(?:people|persons?|visitors?)|its hours?|its timings?|their timings?)\b",
        re.I,
    )
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content", ""))
        if topic.search(content):
            return content.casefold()
        if not followup.search(content):
            break
    return ""


def _answers_service_hours_clarification(
    message: str, history: list[dict[str, Any]]
) -> bool:
    answer = message.casefold().strip(" .!?/")
    if answer not in {"general", "general ward", "general inpatient ward", "general inpatient wards"}:
        return False
    last_assistant = next(
        (
            str(turn.get("content", "")).casefold()
            for turn in reversed(history)
            if turn.get("role") == "assistant"
        ),
        "",
    )
    return "which department or hospital service" in last_assistant

def _extract_time_input(message: str) -> str | None:
    pattern = r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?\b"
    for match in re.finditer(pattern, message, re.I):
        value = match.group(0).strip()
        if ":" in value or re.search(r"(?:am|pm|a\.m\.|p\.m\.)$", value, re.I):
            return value
    return None


def _find_weekday(message: str) -> str | None:
    """Recognize weekday names, including common one-letter typos."""
    weekdays = list(calendar.day_name)
    for word in re.findall(r"[a-z]+", message.casefold()):
        for weekday in weekdays:
            if word == weekday.casefold():
                return weekday

        if len(word) < 6:
            continue

        closest = max(
            weekdays,
            key=lambda day: SequenceMatcher(None, word, day.casefold()).ratio(),
        )
        similarity = SequenceMatcher(None, word, closest.casefold()).ratio()
        if similarity >= 0.86:
            return closest

    return None


def _requested_specialization(message: str) -> str | None:
    normalized = normalize_specialization(message)
    known_specialties = {
        "cardiologist", "dermatologist", "neurologist", "orthopedic specialist",
        "ent specialist", "pediatrician", "general physician", "gynecologist", "physiotherapist",
    }
    if normalized and normalized.lower() in known_specialties:
        return normalized

    specialty_match = re.search(
        r"\b(?:need|want|find|looking for|do you have|is there|show me)\s+"
        r"(?:(?:an|a|the)\b\s*)?([a-z][a-z -]*?)"
        r"(?:\s+(?:doctor|specialist|available|appointment)\b|[?!.]|$)",
        message,
        re.I,
    )
    if not specialty_match:
        return None
    requested = specialty_match.group(1).strip()
    requested = re.sub(r"^(?:another|any|some)\s+", "", requested, flags=re.I)
    if requested.lower() in {
        "appointment", "another", "doctor", "specialist", "any", "some", "available",
    }:
        return None
    if set(requested.lower().split()) & {
        "to", "need", "want", "see", "check", "me", "my", "pain", "issue", "problem",
    }:
        return None
    if requested.lower().startswith(("to book", "to make", "to schedule", "to see")):
        return None
    if "appointment" in requested.lower():
        return None
    return normalize_specialization(requested)


def _suggest_specialization(
    message: str,
    requested_specialization: str | None,
) -> str | None:
    doctors = list_doctors()
    available = {doctor["specialization"] for doctor in doctors}
    if requested_specialization and requested_specialization.casefold() in {
        specialization.casefold() for specialization in available
    }:
        return None

    if requested_specialization and any(
        key.casefold() == requested_specialization.casefold()
        and value.casefold() == requested_specialization.casefold()
        for key, value in ALIASES.items()
    ):
        return None

    options = {specialization.casefold(): specialization for specialization in available}
    options.update({key.casefold(): value for key, value in ALIASES.items()})
    if requested_specialization:
        words = requested_specialization.lower().split()
    else:
        words = re.findall(r"[a-z]+", message.lower())

    best_match = None
    best_score = 0.0
    for word in words:
        if len(word) < 7:
            continue
        for option, specialization in options.items():
            if abs(len(word) - len(option)) > 2:
                continue
            score = SequenceMatcher(None, word, option).ratio()
            if score > best_score:
                best_match = specialization
                best_score = score

    return best_match if best_score >= 0.84 else None


def _patient_fields(extracted: dict[str, Any]) -> dict[str, Any]:
    """Accept either explicit patient_* keys or their shorter local-parser forms."""
    aliases = {
        "patient_name": ("patient_name", "name"),
        "patient_phone": ("patient_phone", "phone"),
        "patient_age": ("patient_age", "age"),
        "patient_weight": ("patient_weight", "weight"),
    }
    fields = {}
    for field, names in aliases.items():
        value = next((extracted[name] for name in names if extracted.get(name) is not None), None)
        if value is not None:
            fields[field] = value
    return fields


def _read_patient_details(
    message: str, *, allow_name: bool = True
) -> dict[str, Any]:
    """Pick up patient fields sent together after the assistant asks for them."""
    phone = extract_phone_number(message)
    age = re.search(r"\b(\d{1,3})\s*(?:years? old|yo|age|,)", message, re.I)
    weight = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b", message, re.I)
    name = re.sub(
        r"(?:\bPAT\d+\b|\+?\d[\d ()-]*\d|\b\d{7}\b|"
        r"\b\d{1,3}\s*(?:years? old|yo|age)\b\s*,?|"
        r"\b\d{1,3}\s*,|\b\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b)",
        "",
        message,
        flags=re.I,
    ).strip(" ,.;")
    name = re.sub(r"\b(?:years? old|yo|age|kg|kgs|kilograms?)\b", "", name, flags=re.I)
    name = re.sub(r"\s*[,;]\s*", " ", name).strip()

    fields = {}
    if phone:
        fields["patient_phone"] = phone
    if age:
        fields["patient_age"] = int(age.group(1))
    if weight:
        fields["patient_weight"] = float(weight.group(1))
    if allow_name and name and not re.fullmatch(r"(?:\d+|kg|years? old)", name, re.I):
        fields["patient_name"] = name
    return fields


def understand_intent(state: dict[str, Any]) -> dict[str, Any]:
    message = state.get("user_message", "").strip()
    classification_message = _strip_leading_list_number(message)
    lower_text = classification_message.lower()
    pending_action = state.get("pending_action")
    pending_doctor_offer = pending_action == "offer_doctor_directory"
    pending_confirmation = pending_action == "confirm_specialization"

    answer = classification_message.casefold().strip(" .!?,")
    yes_answers = {
        "okay", "ok", "yes", "yes please", "sure", "please",
        "that would help", "yeah", "yep", "correct",
        "that's right", "that is right",
    }
    no_answers = {
        "no", "no thanks", "no thank you", "nope", "not that",
    }

    if (
        pending_action == "time"
        and state.get("appointment_time_ambiguous")
        and answer in {"am", "a.m", "a.m.", "pm", "p.m", "p.m."}
    ):
        meridiem = "AM" if answer.startswith("a") else "PM"
        hour = state.get("appointment_time_hour")
        if hour:
            return {
                "intent": "continue_booking",
                "appointment_time": f"{hour} {meridiem}",
                "appointment_time_ambiguous": False,
                "appointment_time_hour": None,
                "response_data": None,
                "response_type": "text",
            }

    if pending_doctor_offer:
        if answer in yes_answers:
            return {
                "intent": "list_doctors",
                "pending_action": None,
                "response_data": None,
                "response_type": "text",
            }
        if answer in no_answers:
            return {
                "intent": "decline_offer",
                "pending_action": None,
                "response_data": None,
                "response_type": "text",
            }

    if pending_confirmation:
        if answer in yes_answers:
            return {
                "intent": "find_doctor",
                "specialization": state.get("suggested_specialization"),
                "suggested_specialization": None,
                "selected_doctor": None,
                "appointment_date": None,
                "appointment_time": None,
                "doctors": [],
                "pending_action": None,
                "response_data": None,
                "response_type": "text",
            }
        if answer in no_answers:
            return {
                "intent": "find_doctor",
                "specialization": None,
                "suggested_specialization": None,
                "pending_action": None,
                "response_data": None,
                "response_type": "text",
            }

    local_extraction = understand_locally(classification_message)
    has_list_prefix = classification_message != message
    local_intent = local_extraction["intent"]

    history = state.get("conversation_history", [])
    if local_intent == "unclear_message" and (
        _is_contextual_hospital_followup(lower_text, history)
        or _answers_service_hours_clarification(classification_message, history)
    ):
        # Classify contextual references before graph routing; keep the actual
        # user message untouched so the knowledge node can resolve its context.
        local_intent = "general_hospital_query"
        local_extraction["intent"] = local_intent

    patient_id_match = re.search(r"\bPAT\d{3,}\b", classification_message, re.I)
    if patient_id_match and local_intent != "check_patient_details":
        local_intent = "check_appointment"
        local_extraction["intent"] = local_intent

    patient_details = _read_patient_details(
        classification_message, allow_name=pending_action == "patient_info"
    )
    has_lookup_credential = (
        pending_action in {"lookup_appointment", "lookup_patient"}
        and bool(patient_id_match or patient_details.get("patient_phone"))
    )
    has_patient_details = pending_action == "patient_info" and any(
        patient_details.get(field) is not None
        for field in (
            "patient_name",
            "patient_phone",
            "patient_age",
            "patient_weight",
        )
    )

    if pending_action == "patient_info" and has_patient_details:
        return {
            "intent": "continue_booking",
            **patient_details,
            "response_data": None,
            "response_type": "text",
            "pending_action": "patient_info",
        }

    # A user can restate their intention to book while already in a booking
    # step. Keep the selected doctor/date/time and continue from that step.
    continuing_booking = (
        pending_action in {"date", "time", "patient_info"}
        and local_intent == "book_appointment"
        and state.get("selected_doctor") is not None
        and not local_extraction.get("specialization")
    )
    if continuing_booking:
        local_intent = "continue_booking"
        local_extraction["intent"] = "continue_booking"

    simple_intents = {
        "general_chat",
        "help",
        "thanks",
        "list_doctors",
        "privacy_boundary",
        "check_appointment",
        "check_patient_details",
        "cancel_appointment",
        "confirm_specialization",
        "symptom_guidance",
        "general_hospital_query",
        "continue_booking",
    }

    if (
        local_intent in simple_intents
        or patient_id_match
        or has_patient_details
        or has_lookup_credential
        or (
            local_intent == "find_doctor"
            and not local_extraction.get("specialization")
        )
    ):
        extracted = local_extraction.copy()
    else:
        extracted = understand_with_llm(classification_message, history) or local_extraction.copy()

    if has_patient_details:
        extracted.update(patient_details)

    for field in (
        "specialization",
        "suggested_specialization",
        "weekday",
        "patient_name",
        "patient_phone",
        "patient_age",
        "patient_weight",
        "patient_id",
        "lookup_patient_id",
        "lookup_patient_name",
    ):
        if local_extraction.get(field) is not None:
            extracted[field] = local_extraction[field]

    if local_extraction.get("weekday"):
        extracted["weekday"] = local_extraction["weekday"]

    if local_intent in simple_intents:
        extracted["intent"] = local_intent
    elif local_intent in {"book_appointment", "find_doctor"}:
        extracted["intent"] = local_intent

    specialization = normalize_specialization(
        extracted.get("specialization")
    )

    if local_intent == "confirm_specialization":
        specialization = None

    intent = extracted.get("intent") or "unclear_message"

    updates: dict[str, Any] = {
        "intent": intent,
        "response_data": None,
        "response_type": "text",
        "resolved_question": (
            extracted.get("resolved_question")
            if local_intent == "unclear_message" and intent == "general_hospital_query"
            else None
        ),
    }

    if local_intent == "confirm_specialization":
        updates.update(
            specialization=None,
            suggested_specialization=local_extraction.get(
                "suggested_specialization"
            ),
            pending_action="confirm_specialization",
        )

    if specialization:
        updates["specialization"] = specialization

    appointment_date = (
        extracted.get("date")
        or extracted.get("appointment_date")
        or local_extraction.get("appointment_date")
    )

    weekday = extracted.get("weekday") or local_extraction.get("weekday")

    if appointment_date:
        updates["appointment_date"] = appointment_date
    elif weekday and (
        pending_action in {"doctor", "date", "time"}
        or intent == "book_appointment"
    ):
        updates["appointment_date"] = (
            _next_date_for_weekday(weekday).isoformat()
        )

    appointment_time = (
        extracted.get("time")
        or extracted.get("appointment_time")
        or local_extraction.get("appointment_time")
    )
    if appointment_time:
        updates["appointment_time"] = appointment_time

    bare_hour = ambiguous_hour_from_text(classification_message)
    booking_turn = (
        pending_action in {"date", "time"}
        or intent in {"book_appointment", "continue_booking"}
    )
    if bare_hour and booking_turn:
        updates["appointment_time"] = None
        updates["appointment_time_ambiguous"] = True
        updates["appointment_time_hour"] = bare_hour
    elif appointment_time and re.search(r"(?:am|pm|a\.m\.|p\.m\.)", classification_message, re.I):
        updates["appointment_time_ambiguous"] = False
        updates["appointment_time_hour"] = None

    updates.update(_patient_fields(extracted))

    lookup_intent = intent in {"check_appointment", "check_patient_details"}
    lookup_pending = pending_action in {"lookup_appointment", "lookup_patient"}
    if lookup_intent or lookup_pending:
        submitted_id = patient_id_match.group(0).upper() if patient_id_match else None
        submitted_phone = (
            extracted.get("patient_phone") or patient_details.get("patient_phone")
        )
        updates["lookup_patient_id"] = submitted_id or (
            state.get("lookup_patient_id") if lookup_pending else None
        )
        updates["lookup_patient_phone"] = submitted_phone or (
            state.get("lookup_patient_phone") if lookup_pending else None
        )
        if submitted_id or submitted_phone:
            updates["lookup_verified_patient_id"] = None
        elif state.get("lookup_verified_patient_id"):
            updates["lookup_verified_patient_id"] = state["lookup_verified_patient_id"]
        updates["patient_id"] = None

    if intent == "check_appointment":
        if extracted.get("patient_name"):
            updates["lookup_patient_name"] = extracted["patient_name"]
        elif extracted.get("patient_phone"):
            updates["lookup_patient_name"] = None

    if patient_id_match and not (lookup_intent or lookup_pending):
        patient_id = patient_id_match.group(0).upper()
        updates["patient_id"] = patient_id

        if local_intent != "check_patient_details":
            updates["intent"] = "check_appointment"

        patient = get_patient(patient_id)
        if patient:
            updates.update(
                patient_name=patient["name"],
                patient_phone=patient["phone"],
                patient_age=patient["age"],
                patient_weight=patient["weight"],
            )
    elif (
        pending_action in {"lookup_appointment", "lookup_patient"}
        and extracted.get("patient_phone")
    ):
        updates["patient_id"] = None
    elif (
        extracted.get("patient_phone")
        and updates["intent"] == "check_appointment"
    ):
        updates["patient_id"] = None

    if pending_action == "lookup_appointment" and updates["intent"] not in {
        "general_chat",
        "help",
        "thanks",
        "list_doctors",
        "privacy_boundary",
        "symptom_guidance",
        "general_hospital_query",
        "cancel_appointment",
        "find_doctor",
        "book_appointment",
    }:
        updates["intent"] = "check_appointment"

    elif pending_action == "lookup_patient" and updates["intent"] not in {
        "general_chat",
        "help",
        "thanks",
        "list_doctors",
        "privacy_boundary",
        "symptom_guidance",
        "general_hospital_query",
        "cancel_appointment",
        "find_doctor",
        "book_appointment",
    }:
        updates["intent"] = "check_patient_details"

    unrelated_intents = {
        "general_chat",
        "help",
        "thanks",
        "list_doctors",
        "privacy_boundary",
        "symptom_guidance",
        "general_hospital_query",
        "cancel_appointment",
        "check_appointment",
        "check_patient_details",
        "find_doctor",
        "book_appointment",
        "confirm_specialization",
        "unclear_message",
    }

    if (
        pending_action in {"doctor", "date", "time"}
        and updates["intent"] not in unrelated_intents
    ):
        updates["intent"] = "continue_booking"

    # Short answers are useful booking continuations even when they are not
    # meaningful intents on their own. Explicit unrelated intents still route
    # to their own nodes and leave the pending booking available to resume.
    if (
        pending_action == "date"
        and (appointment_date or weekday)
        and updates["intent"] not in {
            "general_chat", "help", "thanks", "list_doctors",
            "privacy_boundary", "symptom_guidance", "general_hospital_query",
            "cancel_appointment", "check_appointment", "check_patient_details",
        }
    ):
        updates["intent"] = "continue_booking"
    elif (
        pending_action == "time"
        and (appointment_time or bare_hour or state.get("appointment_time_ambiguous"))
        and updates["intent"] not in {
            "general_chat", "help", "thanks", "list_doctors",
            "privacy_boundary", "symptom_guidance", "general_hospital_query",
            "cancel_appointment", "check_appointment", "check_patient_details",
        }
    ):
        updates["intent"] = "continue_booking"

    if (
        not specialization
        and state.get("specialization")
        and updates["intent"] in {
            "continue_booking",
            "book_appointment",
        }
    ):
        updates["specialization"] = state["specialization"]

    if (
        patient_id_match
        and local_intent != "check_patient_details"
    ):
        updates["intent"] = "check_appointment"

    if (
        updates["intent"] in {"check_appointment", "privacy_boundary"}
        or pending_action == "lookup_appointment"
    ):
        doctor_name = _requested_doctor_name(classification_message)
        if doctor_name:
            updates["appointment_doctor_filter"] = doctor_name
        elif (
            pending_action == "lookup_appointment"
            and specialization
        ):
            updates["appointment_specialty_filter"] = specialization

    doctors = state.get("doctors", [])
    can_select_doctor = (
        pending_action in {"doctor", "date", "time"}
        or updates["intent"] in {"find_doctor", "book_appointment"}
    )

    selected = (
        _match_doctor(classification_message, doctors)
        if (
            can_select_doctor
            and updates["intent"] != "check_appointment"
            and not has_list_prefix
        )
        else None
    )

    if selected:
        updates.update(
            selected_doctor=selected,
            pending_action="date",
            intent="continue_booking",
        )
    elif pending_action == "doctor" and doctors:
        selection = re.fullmatch(r"(?:first|second|[1-9]\d*)", answer)
        if selection:
            choice = selection.group(0)
            index = (
                0
                if choice in {"first", "1"}
                else 1
                if choice == "second"
                else int(choice) - 1
            )
            if index < len(doctors):
                updates.update(
                    selected_doctor=doctors[index],
                    pending_action="date",
                    intent="continue_booking",
                )

    is_new_flow = updates["intent"] in {
        "find_doctor",
        "book_appointment",
    }

    if is_new_flow:
        reuse_selected_doctor = bool(
            updates["intent"] == "book_appointment"
            and state.get("selected_doctor")
            and (
                not specialization
                or re.search(
                    r"\b(?:same doctor|same one|same physician)\b",
                    classification_message,
                    re.I,
                )
            )
        )

        updates["appointment_date"] = updates.get("appointment_date")
        updates["appointment_time"] = updates.get("appointment_time")
        if not updates["appointment_time"]:
            updates["appointment_time_ambiguous"] = False
            updates["appointment_time_hour"] = None

        if reuse_selected_doctor:
            updates["specialization"] = state.get("specialization")
            updates["selected_doctor"] = state.get("selected_doctor")
        else:
            updates["selected_doctor"] = None
            updates["specialization"] = specialization
            updates["doctors"] = []

        updates["pending_action"] = None

    if pending_confirmation or pending_doctor_offer:
        updates["pending_action"] = None
        updates["suggested_specialization"] = None

    # Verification inputs are only relevant to the lookup flow. An explicit
    # new task must not inherit a previous patient's ID/phone in checkpointed
    # state, while a submitted ID or phone remains a valid lookup continuation.
    if (
        pending_action in {"lookup_appointment", "lookup_patient"}
        and updates["intent"] not in {"check_appointment", "check_patient_details"}
    ):
        updates.update(
            pending_action=None,
            lookup_patient_id=None,
            lookup_patient_phone=None,
            lookup_patient_name=None,
            lookup_verified_patient_id=None,
            appointment_doctor_filter=None,
            appointment_specialty_filter=None,
            patient_id=None,
        )

    return updates


def _match_doctor(message: str, doctors: list[dict[str, Any]]) -> dict[str, Any] | None:
    choice = message.strip().lower()
    for doctor in doctors:
        name_without_title = doctor["name"].removeprefix("Dr. ").lower()
        if (
            doctor["name"].lower() in choice
            or name_without_title in choice
            or doctor["id"].lower() == choice
            or (len(choice) >= 3 and choice in name_without_title)
        ):
            return doctor
    return None


def _requested_doctor_name(message: str) -> str | None:
    lower_message = message.casefold()
    for doctor in list_doctors():
        name = doctor.get("name", "")
        without_title = name.removeprefix("Dr. ")
        if name.casefold() in lower_message or without_title.casefold() in lower_message:
            return name
    return None


def find_doctor_node(state: dict[str, Any]) -> dict[str, Any]:
    specialization = state.get("specialization")
    if not specialization:
        if state.get("pending_action") == "doctor" and state.get("doctors"):
            return {
                "response": "Which doctor would you like to see? You can choose one from the list above.",
                "response_type": "text",
            }

        message = state.get("user_message", "").casefold()
        mentions_back_pain = bool(re.search(r"\bback\s?pain\b|\bbackache\b", message))
        asks_for_doctor = any(
            phrase in message
            for phrase in ("doctor", "check me", "someone available", "see someone")
        )
        if mentions_back_pain and asks_for_doctor:
            doctors = list_doctors()
            return {
                "doctors": doctors,
                "response": (
                    "I'm sorry your back is bothering you. I can't assess symptoms in chat, "
                    "but I can show you the doctors currently listed at this hospital. "
                    "Choose one and I'll help you check appointment times."
                ),
                "response_type": "doctor_list",
                "response_data": doctors,
                "pending_action": "doctor",
            }
        return {
            "response": "Which type of doctor are you looking for? For example, a cardiologist or skin doctor.",
            "response_type": "text",
            "pending_action": None,
        }

    doctors = find_doctors(specialization)
    if not doctors:
        article = "an" if specialization[:1].lower() in "aeiou" else "a"
        return {
            "response": (
                f"We don't currently have {article} {specialization.lower()} available. "
                "I can show you the doctors we do have, if that would help."
            ),
            "response_type": "text",
            "pending_action": "offer_doctor_directory",
            "doctors": [],
        }

    names = ", ".join(doctor["name"] for doctor in doctors)
    return {
        "doctors": doctors,
        "response": f"I found {len(doctors)} {specialization.lower()}s: {names}. Which doctor would you like?",
        "response_type": "doctor_list",
        "response_data": doctors,
        "pending_action": "doctor",
    }


def list_doctors_node(state: dict[str, Any]) -> dict[str, Any]:
    doctors = list_doctors()
    specialties = sorted({doctor["specialization"] for doctor in doctors})
    summary = ", ".join(specialties)
    return {
        "doctors": doctors,
        "response": (
            f"We have {len(doctors)} doctors across {len(specialties)} specialties: "
            f"{summary}. Here are the doctors on our team."
        ),
        "response_type": "doctor_list",
        "response_data": doctors,
        "selected_doctor": None,
        "specialization": None,
        "appointment_date": None,
        "appointment_time": None,
        "pending_action": "doctor",
    }


def conversational_response(state: dict[str, Any]) -> dict[str, Any]:
    intent = state.get("intent")
    message = state.get("user_message", "").lower()

    if intent == "symptom_guidance":
        if any(
            term in message
            for term in (
                "chest pain",
                "difficulty breathing",
                "shortness of breath",
                "can't breathe",
                "cannot breathe",
                "trouble breathing",
            )
        ):
            response = (
                "These symptoms may indicate a medical emergency. "
                "Call your local emergency service or go to the nearest "
                "emergency department now. Do not wait for an appointment."
            )
        elif any(
            term in message
            for term in (
                "can you diagnose",
                "diagnose my symptoms",
                "give me a diagnosis",
                "what is my diagnosis",
            )
        ):
            response = (
                "I can't diagnose medical conditions. I can provide "
                "general health information and help you find an appropriate "
                "doctor for a professional assessment."
            )
        elif "headache" in message or "migraine" in message:
            response = (
                "Headaches can have several causes, and I can't diagnose "
                "the cause here. If your headache is sudden and severe, or "
                "comes with weakness, confusion, vision loss, or difficulty "
                "speaking, seek emergency medical care. Otherwise, I can "
                "help you find an appropriate doctor."
            )
        elif "knee pain" in message:
            response = (
                "An orthopedic specialist may be a suitable starting point "
                "for knee pain. I can't diagnose the cause, but I can help "
                "you find an appropriate doctor."
            )
        else:
            response = (
                "I can't diagnose symptoms. Tell me a little more about "
                "what you're experiencing, or I can help you find an "
                "appropriate doctor."
            )

        return {
            "response": response,
            "response_type": "text",
            "pending_action": None,
        }

    if intent == "general_chat":
        if any(
            phrase in message
            for phrase in ("how are you", "what's up", "what is up")
        ):
            response = (
                "I'm here and ready to help. Are you looking for a doctor, "
                "appointment assistance, or hospital information?"
            )
        else:
            response = (
                "Hello! How can I help you today? I can assist with doctors, "
                "appointments, and hospital services."
            )

        return {
            "response": response,
            "response_type": "text",
        }

    if intent == "unclear_message":
        if re.search(r"\b(?:hours?|timings?|times?)\b", message):
            return {
                "response": "Which department or hospital service do you mean? Please name it so I can check its hours.",
                "response_type": "text",
                "pending_action": None,
            }
        return {
            "response": (
                "I didn't quite understand that. You can ask me about "
                "doctors, appointments, or hospital services."
            ),
            "response_type": "text",
            "pending_action": None,
        }

    replies = {
        "greeting": "Hello! How can I help you today?",
        "confirm_specialization": (
            f'Did you mean "{state.get("suggested_specialization")}"? '
            "Reply yes to see doctors in that specialty."
        ),
        "help": (
            "I can help you find a doctor, check appointment availability, "
            "book a visit, or look up your patient and appointment details. "
            "What would you like help with?"
        ),
        "thanks": "You're welcome. Let me know if you need anything else.",
        "decline_offer": (
            "No problem. I can help you find a doctor whenever you're ready."
        ),
    }

    return {
        "response": replies.get(
            intent,
            general_response(state)["response"],
        ),
        "response_type": "text",
    }



def continue_booking(state: dict[str, Any]) -> dict[str, Any]:
    doctor = state.get("selected_doctor")
    if not doctor:
        return find_doctor_node(state)

    appointment_date = state.get("appointment_date")
    if not appointment_date:
        requested_day = _find_weekday(state.get("user_message", ""))
        if requested_day:
            weekday = _next_working_weekday(requested_day, doctor["working_days"])
            if not weekday:
                days = ", ".join(doctor["working_days"])
                return {
                    "response": (
                        f"{doctor['name']} doesn't work on {requested_day}. "
                        f"They are available on {days}. Which day would suit you?"
                    ),
                    "response_type": "text",
                    "pending_action": "date",
                }
            appointment_date = _next_date_for_weekday(weekday).isoformat()
        else:
            return _ask_for_date(doctor)

    appointment_time = state.get("appointment_time") or _extract_time_input(
        state.get("user_message", "")
    )

    def keep_booking_details(response: dict[str, Any]) -> dict[str, Any]:
        response["appointment_date"] = appointment_date
        if appointment_time and "appointment_time" not in response:
            response["appointment_time"] = appointment_time
        return response

    try:
        parsed_date = date.fromisoformat(appointment_date)
    except ValueError:
        return keep_booking_details({
            "response": "That date doesn't look valid. Please use YYYY-MM-DD.",
            "response_type": "error",
            "pending_action": "date",
        })

    if parsed_date < date.today():
        return keep_booking_details({
            "response": "That date has already passed. Please choose a future date.",
            "response_type": "error",
            "pending_action": "date",
        })

    if parsed_date.strftime("%A") not in doctor["working_days"]:
        days = ", ".join(doctor["working_days"])
        return keep_booking_details({
            "response": f"{doctor['name']} doesn't work on {parsed_date.strftime('%A')}. Working days are {days}. Please choose another date.",
            "response_type": "error",
            "pending_action": "date",
        })

    if state.get("appointment_time_ambiguous") and state.get("appointment_time_hour"):
        hour = state["appointment_time_hour"]
        return {
            "response": (
                f"Did you mean {hour} AM or {hour} PM? "
                "Please specify AM or PM before I check availability."
            ),
            "response_type": "text",
            "pending_action": "time",
            "appointment_date": appointment_date,
            "appointment_time": None,
            "appointment_time_ambiguous": True,
            "appointment_time_hour": hour,
        }

    if not appointment_time:
        slots = get_available_slots(doctor["id"], appointment_date)
        if not slots:
            return keep_booking_details({
                "response": f"All appointment times on {appointment_date} are booked. Please choose another date.",
                "response_type": "error",
                "pending_action": "date",
            })
        return keep_booking_details({
            "response": f"Available times on {appointment_date}: {_format_times(slots)}. Which time would you like?",
            "response_type": "appointment_slots",
            "response_data": {"doctor": doctor, "date": appointment_date, "slots": slots},
            "pending_action": "time",
        })

    if appointment_time not in doctor["appointment_slots"]:
        appointment_time = normalize_time_input(
            appointment_time, doctor["appointment_slots"]
        )
    if not appointment_time:
        slots = get_available_slots(doctor["id"], appointment_date)
        return keep_booking_details({
            "response": f"Please choose one of these available times: {_format_times(slots)}.",
            "response_type": "appointment_slots",
            "response_data": {"doctor": doctor, "date": appointment_date, "slots": slots},
            "pending_action": "time",
        })

    availability = check_availability(doctor["id"], appointment_date, appointment_time)
    if not availability["available"]:
        slots = get_available_slots(doctor["id"], appointment_date)
        if "already booked" in availability["message"].lower():
            explanation = "That time is already booked."
        else:
            explanation = "That isn't one of this doctor's appointment times."
        if slots:
            message = (
                f"{explanation} Here are the available times for "
                f"{appointment_date}: {_format_times(slots)}. Which would you prefer?"
            )
            response_type = "appointment_slots"
            data = {"doctor": doctor, "date": appointment_date, "slots": slots}
            pending_action = "time"
        else:
            message = f"There are no appointment times available on {appointment_date}. Please choose another date."
            response_type = "error"
            data = None
            pending_action = "date"
        return keep_booking_details({
            "appointment_time": None,
            "response": message,
            "response_type": response_type,
            "response_data": data,
            "pending_action": pending_action,
        })

    if state.get("patient_id"):
        patient = get_patient(state["patient_id"])
    elif state.get("patient_phone"):
        patient = find_patient_by_phone(state["patient_phone"])
    else:
        patient = None

    if patient:
        state = {
            **state,
            "patient_id": patient["patient_id"],
            "patient_name": patient["name"],
            "patient_phone": patient["phone"],
            "patient_age": patient["age"],
            "patient_weight": patient["weight"],
        }

    booking_state = {
        **state,
        "appointment_date": appointment_date,
        "appointment_time": appointment_time,
    }

    missing_fields = _missing_patient_fields(state)
    if missing_fields:
        return keep_booking_details({
            "response": _patient_info_prompt(missing_fields),
            "response_type": "text",
            "pending_action": "patient_info",
        })

    return _save_booking(booking_state, doctor, appointment_date, appointment_time)


def _format_times(slots: list[str]) -> str:
    return ", ".join(format_time(slot) for slot in slots)


def _ask_for_date(doctor: dict[str, Any]) -> dict[str, Any]:
    days = ", ".join(doctor["working_days"])
    return {
        "response": f"{doctor['name']} works on {days}. Which date would you prefer? (YYYY-MM-DD)",
        "response_type": "appointment_slots",
        "response_data": {
            "doctor": doctor,
            "working_days": doctor["working_days"],
            "slots": doctor["appointment_slots"],
        },
        "pending_action": "date",
    }


def _next_working_weekday(message: str, working_days: list[str]) -> str | None:
    choice = message.strip().lower()
    return next((day for day in working_days if day.lower() == choice), None)


def _next_date_for_weekday(weekday: str) -> date:
    weekday_number = list(calendar.day_name).index(weekday)
    days_ahead = (weekday_number - date.today().weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return date.today() + timedelta(days=days_ahead)


def _missing_patient_fields(state: dict[str, Any]) -> list[str]:
    labels = {
        "patient_name": "full name",
        "patient_phone": "phone number",
        "patient_age": "age",
        "patient_weight": "weight in kg",
    }
    return [label for field, label in labels.items() if state.get(field) is None]


def _patient_info_prompt(missing_fields: list[str]) -> str:
    return (
        "The slot is available. Please provide your "
        + ", ".join(missing_fields)
        + ". You can send them together (for example: Ali Raza, 03001234567, "
        "30 years old, 72 kg)."
    )


def _save_booking(
    state: dict[str, Any],
    doctor: dict[str, Any],
    appointment_date: str,
    appointment_time: str,
) -> dict[str, Any]:
    try:
        patient = get_patient(state["patient_id"]) if state.get("patient_id") else None
        if not patient:
            patient = save_patient(
                state["patient_name"],
                state["patient_phone"],
                state["patient_age"],
                state["patient_weight"],
            )
        appointment = book_appointment(
            patient["patient_id"], doctor["id"], appointment_date, appointment_time
        )
    except (ValueError, ValidationError) as error:
        return {
            "response": str(error),
            "response_type": "error",
            "pending_action": "patient_info",
        }

    return {
        "patient_id": patient["patient_id"],
        "pending_action": None,
        "response": (
            f"Your appointment is confirmed with {doctor['name']} on "
            f"{appointment['date']} at {format_time(appointment['time'])}. "
            f"Your patient ID is {patient['patient_id']}."
        ),
        "response_type": "appointment_confirmation",
        "response_data": {
            "appointment": appointment,
            "patient": {"patient_id": patient["patient_id"]},
        },
    }


def get_appointment_node(state: dict[str, Any]) -> dict[str, Any]:
    patient, verification = _verify_lookup_identity(state)
    if not patient:
        if verification == "need_phone":
            response = "Please enter the registered phone number for verification."
        elif verification == "need_id":
            response = "Please enter your patient ID to complete verification."
        elif verification == "invalid":
            response = (
                "I couldn't verify those details. Please start again with your patient ID "
                "and the registered phone number."
            )
        else:
            response = (
                "For privacy, please verify your identity with both your patient ID and "
                "the phone number registered when booking. A name or either item alone "
                "isn't enough to access records."
            )
        return {
            "response": response,
            "response_type": "text",
            "pending_action": "lookup_appointment",
            "lookup_verified_patient_id": None,
            "patient_id": None,
            **({"lookup_patient_id": None, "lookup_patient_phone": None}
               if verification == "invalid" else {}),
        }

    appointments = get_patient_appointments(patient["patient_id"])
    doctor_filter = state.get("appointment_doctor_filter")
    specialty_filter = state.get("appointment_specialty_filter")
    if doctor_filter:
        appointments = [
            appointment for appointment in appointments
            if appointment.get("doctor_name", "").casefold() == doctor_filter.casefold()
        ]
    if specialty_filter:
        appointments = [
            appointment for appointment in appointments
            if appointment.get("specialization", "").casefold() == specialty_filter.casefold()
        ]
    if not appointments:
        requested_provider = doctor_filter or specialty_filter
        return {
            "response": (
                f"I couldn't find an appointment with {requested_provider} linked to your record."
                if requested_provider else "I found your patient record, but there are no appointments linked to it."
            ),
            "response_type": "text",
            "pending_action": None,
            "appointment_doctor_filter": None,
            "appointment_specialty_filter": None,
            "lookup_patient_id": None,
            "lookup_patient_phone": None,
            "lookup_verified_patient_id": None,
            "patient_id": None,
        }

    summaries = [
        f"{appointment['appointment_id']}: {appointment['doctor_name']} "
        f"({appointment['specialization']}) — "
        f"{appointment['date']} at {format_time(appointment['time'])}, "
        f"{appointment['status']}."
        for appointment in appointments
    ]
    return {
        "response": "Here are your appointments: " + " ".join(summaries),
        "response_type": "appointment_details",
        "response_data": appointments,
        "pending_action": None,
        "appointment_doctor_filter": None,
        "appointment_specialty_filter": None,
        "lookup_patient_name": None,
        "lookup_patient_id": None,
        "lookup_patient_phone": None,
        "lookup_verified_patient_id": None,
        "patient_id": None,
    }


def get_patient_details_node(state: dict[str, Any]) -> dict[str, Any]:
    patient, verification = _verify_lookup_identity(state)
    if not patient:
        if verification == "need_phone":
            response = "Please enter the registered phone number for verification."
        elif verification == "need_id":
            response = "Please enter your patient ID to complete verification."
        elif verification == "invalid":
            response = (
                "I couldn't verify those details. Please start again with your patient ID "
                "and the registered phone number."
            )
        else:
            response = (
                "For privacy, please verify your identity with both your patient ID and "
                "the phone number registered when booking. A name or either item alone "
                "isn't enough to access records."
            )
        return {
            "response": response,
            "response_type": "text",
            "pending_action": "lookup_patient",
            "lookup_verified_patient_id": None,
            "patient_id": None,
            **({"lookup_patient_id": None, "lookup_patient_phone": None}
               if verification == "invalid" else {}),
        }

    return {
        "patient_id": None,
        "response": (
            f"Patient record for {patient['name']}: phone {patient['phone']}, "
            f"age {patient['age']}, weight {patient['weight']} kg. "
            f"Patient ID: {patient['patient_id']}."
        ),
        "response_type": "patient_details",
        "response_data": patient,
        "pending_action": None,
        "lookup_patient_id": None,
        "lookup_patient_phone": None,
        "lookup_verified_patient_id": None,
    }


def _verify_lookup_identity(
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Require both independent lookup inputs before returning patient data."""
    verified_id = state.get("lookup_verified_patient_id")
    if verified_id:
        patient = get_patient(verified_id)
        if patient:
            return patient, "verified"

    patient_id = state.get("lookup_patient_id")
    phone = state.get("lookup_patient_phone")
    if not patient_id and not phone:
        return None, "need_both"
    if patient_id and not phone:
        return None, "need_phone"
    if phone and not patient_id:
        return None, "need_id"

    try:
        phone = normalize_phone(phone)
        patient = get_patient(patient_id)
        phone_patient = find_patient_by_phone(phone)
    except (TypeError, ValueError):
        return None, "invalid"
    if not patient or not phone_patient or patient["patient_id"] != phone_patient["patient_id"]:
        return None, "invalid"
    return patient, "verified"


def general_response(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("intent") == "privacy_boundary":
        return {
            "response": (
                "For privacy, I can only look up appointments linked to your patient ID or "
                "registered phone number. I can't share a doctor's full appointment list or other patients' details."
            ),
            "response_type": "text",
            "pending_action": None,
        }
    return {
        "response": (
            "I can help with finding doctors, checking availability, appointments, "
            "and patient records. I can't provide diagnosis or treatment advice. "
            "For medical guidance, please contact a healthcare professional."
        ),
        "response_type": "text",
        "pending_action": None,
    }
