import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    continue_booking,
    conversational_response,
    find_doctor_node,
    general_response,
    get_appointment_node,
    get_patient_details_node,
    list_doctors_node,
    understand_intent,
)
from app.graph.state import ChatState

logger = logging.getLogger(__name__)

_knowledge_path = Path(__file__).parent / "nodes" / "knowledge.py"
_knowledge_spec = spec_from_file_location(
    "app.graph.knowledge_node",
    _knowledge_path,
)

if _knowledge_spec is None or _knowledge_spec.loader is None:
    raise ImportError(
        f"Could not load the hospital knowledge node at {_knowledge_path}"
    )

_knowledge_module = module_from_spec(_knowledge_spec)
_knowledge_spec.loader.exec_module(_knowledge_module)
answer_hospital_question = _knowledge_module.answer_hospital_question



def route(state: ChatState) -> str:
    intent = state.get("intent", "unclear_message")
    selected_doctor = state.get("selected_doctor")
    pending_action = state.get("pending_action")

    logger.info("Routing chat turn")

    # 1. Handle normal conversation and symptom guidance.
    conversation_intents = {
        "greeting",
        "help",
        "thanks",
        "confirm_specialization",
        "decline_offer",
        "general_chat",
        "symptom_guidance",
        "unclear_message",
        "privacy_boundary",
    }

    if intent in conversation_intents:
        return "conversation"

    # 2. Explicit requests to view the doctor directory.
    if intent == "list_doctors":
        return "list_doctors"

    # 3. Searching for a doctor should always use the doctor-search node.
    if intent == "find_doctor":
        return "find_doctors"

    # 4. Booking without a selected doctor starts doctor selection.
    # Booking with a selected doctor continues the booking process.
    if intent == "book_appointment":
        if selected_doctor:
            return "continue_booking"
        return "find_doctors"

    # 5. Continue an existing booking when the intent classifier
    # explicitly identifies the message as a booking continuation.
    if intent == "continue_booking":
        return "continue_booking"

    # 6. Patient and appointment operations.
    if intent == "check_appointment":
        return "get_appointment"

    if intent == "check_patient_details":
        return "get_patient_details"

    if intent == "cancel_appointment":
        return "cancel"

    # 7. Hospital policies, visiting hours, and other hospital facts.
    if intent == "general_hospital_query":
        return "knowledge"

    # 8. Fallback for unexpected or unsupported intents.
    logger.warning("No specific route matched; using the general response.")
    return "general"


def cancellation(state: ChatState) -> dict:
    return {
        "response": (
            "Appointment cancellation isn't available yet. "
            "Please contact the hospital desk for help."
        ),
        "response_type": "text",
        "pending_action": None,
    }


def build_graph():
    workflow = StateGraph(ChatState)

    workflow.add_node("understand_intent", understand_intent)
    workflow.add_node("find_doctors", find_doctor_node)
    workflow.add_node("continue_booking", continue_booking)
    workflow.add_node("get_appointment", get_appointment_node)
    workflow.add_node("get_patient_details", get_patient_details_node)
    workflow.add_node("list_doctors", list_doctors_node)
    workflow.add_node("conversation", conversational_response)
    workflow.add_node("cancel", cancellation)
    workflow.add_node("general", general_response)
    workflow.add_node("knowledge", answer_hospital_question)

    workflow.add_edge(START, "understand_intent")

    routes = {
        "find_doctors": "find_doctors",
        "continue_booking": "continue_booking",
        "get_appointment": "get_appointment",
        "get_patient_details": "get_patient_details",
        "list_doctors": "list_doctors",
        "conversation": "conversation",
        "cancel": "cancel",
        "general": "general",
        "knowledge": "knowledge",
    }

    workflow.add_conditional_edges(
        "understand_intent",
        route,
        routes,
    )

    terminal_nodes = (
        "find_doctors",
        "continue_booking",
        "get_appointment",
        "get_patient_details",
        "list_doctors",
        "conversation",
        "cancel",
        "general",
        "knowledge",
    )

    for node in terminal_nodes:
        workflow.add_edge(node, END)

    return workflow.compile(checkpointer=MemorySaver())


chat_graph = build_graph()
