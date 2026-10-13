"""LangGraph node for questions about general hospital information."""

import logging
import re
from typing import Any

from app.rag.pipeline import answer_question


logger = logging.getLogger(__name__)


def contextualize_question(question: str, history: list[dict[str, Any]]) -> str:
    """Resolve short follow-ups against the most recent user question."""
    cleaned = question.strip()
    lower = cleaned.casefold()
    user_turns = [
        str(turn.get("content", "")).strip()
        for turn in history
        if turn.get("role") == "user" and str(turn.get("content", "")).strip()
    ]
    last_assistant = next(
        (
            str(turn.get("content", "")).casefold()
            for turn in reversed(history)
            if turn.get("role") == "assistant"
        ),
        "",
    )
    if (
        cleaned.casefold().strip(" .!?/")
        in {"general", "general ward", "general inpatient ward", "general inpatient wards"}
        and "which department or hospital service" in last_assistant
    ):
        return "What are the general ward visiting hours?"

    previous = ""
    topic_pattern = r"\b(?:hospital|icu|intensive care|visiting|visitor|ward|policy|hours|facility|facilities|department|pharmacy|emergency|lab|laboratory)\b"
    followup_pattern = r"\b(?:repeat|say that again|what did you just say|what about|how many|possibilit\w*|two\s+(?:people|persons?|visitors?)|can (?:two|i|we)|and|also|what about)\b"
    for turn in reversed(user_turns):
        if re.search(r"\b(?:repeat|say that again|what did you just say)\b", turn, re.I):
            continue
        if re.search(topic_pattern, turn, re.I):
            previous = turn
            break
        if not re.search(followup_pattern, turn, re.I):
            # Do not resurrect a topic from before a clearly unrelated request.
            break
    if not previous:
        return cleaned

    asks_about_multiple_visitors = (
        re.search(r"\b(?:more than one|two|2|multiple|several)\b", lower)
        and re.search(r"\b(?:people|persons?|visitors?)\b", lower)
        and re.search(r"\b(?:bring|go|enter|visit|together|inside|in|allow)\b", lower)
        and re.search(r"\bicu\b|intensive care", previous, re.I)
    )
    if asks_about_multiple_visitors:
        return (
            "ICU visitor limit two persons enter together possible exception "
            "main hospital number"
        )

    asks_two_together = bool(
        re.search(r"\b(?:two|2)\s+(?:people|persons?|visitors?)\b", lower)
        and re.search(r"\b(?:possibilit\w*|possible|enter|take|together|allow)\b", lower)
        and re.search(r"\bicu\b|intensive care", previous, re.I)
    )
    if asks_two_together:
        return (
            "ICU visitor limit two persons enter together possible exception "
            "main hospital number"
        )

    if re.search(r"\brepeat\b.*\b(?:that|those|it|hours?|times?)\b", lower) or any(phrase in lower for phrase in (
        "repeat that", "repeat the", "what did you just say", "what did i just ask",
        "visiting hours i just asked", "say that again",
    )):
        return previous

    if re.search(r"\b(?:icu|intensive care|general ward|general inpatient|ward)\b", lower):
        return f"{previous}. Follow-up: {cleaned}"

    if re.search(r"\b(?:its|their)\s+(?:hours?|timings?|times?)\b", lower):
        if re.search(r"\bgeneral ward\b", previous, re.I):
            return f"What are the general ward visiting hours? Follow-up: {cleaned}"
        if re.search(r"\bicu\b|intensive care", previous, re.I):
            return f"What are the ICU visiting hours? Follow-up: {cleaned}"
        return f"{previous}. Follow-up: {cleaned}"

    if re.search(r"\b(?:visitor|visitors|people|person|allowed|enter|maximum|limit|go in)\b", lower):
        ward = "general ward"
        for turn in reversed(user_turns):
            if re.search(r"\bicu\b|intensive care", turn, re.I):
                ward = "ICU"
                break
            if re.search(r"\b(?:general ward|general inpatient)\b", turn, re.I):
                break
        return f"What is the {ward} visitor limit allowed at one time?"

    return cleaned


def answer_hospital_question(state: dict[str, Any]) -> dict[str, Any]:
    question = state.get("user_message", "")
    retrieval_question = state.get("resolved_question") or contextualize_question(
        question, state.get("conversation_history", [])
    )
    try:
        result = answer_question(retrieval_question)
    except Exception as error:
        # Avoid logging the question, retrieved context, or provider exception text.
        logger.error(
            "Could not answer a hospital knowledge question (error_type=%s)",
            type(error).__name__,
        )
        result = {
            "answer": (
                "I can't retrieve hospital information right now. "
                "Please contact the hospital directly for help."
            ),
            "sources": [],
        }

    sources = result.get("sources", [])
    return {
        "response": result.get("answer", "That information is not available."),
        "response_type": "knowledge_answer",
        "response_data": {"sources": sources},
        "sources": sources,
    }
