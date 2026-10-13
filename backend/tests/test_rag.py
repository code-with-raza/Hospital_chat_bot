from pathlib import Path

import pytest

from app.rag import pipeline
from app.rag.retriever import retrieve
from app.rag.reranker import rerank_chunks
from app.rag.vector_stores import collection_size
from app.rag.vector_stores import CHROMA_DB_DIR
from app.graph.graph import _knowledge_module
from app.rag.prompts import HOSPITAL_RAG_SYSTEM_PROMPT


KNOWN_QUESTIONS = [
    (
        "What are the visiting hours?",
        "General inpatient wards visiting hours: 4:00 PM to 7:00 PM daily.",
        1,
        "General inpatient wards are open for visits from 4:00 PM to 7:00 PM daily.",
    ),
    (
        "Is the emergency department open 24 hours?",
        "The Emergency Department is open 24 hours a day, seven days a week.",
        7,
        "Yes. The Emergency Department is open 24 hours a day, seven days a week.",
    ),
    (
        "What should I bring for an appointment?",
        "Bring a valid identification document and relevant previous medical reports when available.",
        1,
        "Bring a valid identification document and previous medical reports if available.",
    ),
    (
        "What are the pharmacy hours?",
        "Emergency and inpatient pharmacy support is available 24 hours according to hospital operational arrangements.",
        3,
        "Emergency and inpatient pharmacy support is available 24 hours.",
    ),
    (
        "What is the hospital privacy policy?",
        "Hospital privacy policy: patient information is confidential and may only be accessed, used, or disclosed for authorized purposes.",
        5,
        "Patient information is confidential and is used or disclosed only for authorized purposes.",
    ),
]


def test_relative_vector_store_path_is_anchored_to_backend_directory():
    expected = Path(__file__).resolve().parents[1] / "app" / "data" / "chroma"

    assert Path(CHROMA_DB_DIR).is_absolute()
    assert Path(CHROMA_DB_DIR) == expected


def test_visitor_limit_followup_retains_icu_policy_context():
    history = [{"role": "user", "content": "What are the ICU visiting hours?"}]
    query = _knowledge_module.contextualize_question("Can two people enter?", history)

    assert "ICU" in query
    assert "visitor limit" in query
    assert "main hospital number" in query
    assert "Do not contradict an explicit" in HOSPITAL_RAG_SYSTEM_PROMPT
    assert "Do not infer whether visitors may take turns" in HOSPITAL_RAG_SYSTEM_PROMPT


def test_ward_change_followup_keeps_prior_topic_and_new_ward():
    query = _knowledge_module.contextualize_question(
        "What about the ICU?",
        [{"role": "user", "content": "What are the general ward visiting hours?"}],
    )
    assert "general ward visiting hours" in query.casefold()
    assert "icu" in query.casefold()


def test_visitor_followup_after_repeat_still_uses_icu_context():
    history = [
        {"role": "user", "content": "What are the ICU visiting hours?"},
        {"role": "assistant", "content": "5 PM to 6 PM daily."},
        {"role": "user", "content": "Can you repeat the visiting hours I just asked about?"},
        {"role": "assistant", "content": "The ICU visiting hours are 5 PM to 6 PM daily."},
    ]
    query = _knowledge_module.contextualize_question("How many visitors are allowed?", history)
    assert "ICU" in query
    assert "visitor limit" in query


def test_unrelated_turn_does_not_resurrect_an_old_hospital_topic():
    query = _knowledge_module.contextualize_question(
        "How many visitors are allowed?",
        [
            {"role": "user", "content": "What are the ICU visiting hours?"},
            {"role": "assistant", "content": "5 PM to 6 PM."},
            {"role": "user", "content": "Book me an appointment with a cardiologist."},
        ],
    )
    assert "ICU" not in query


@pytest.mark.parametrize(
    ("question", "document_text", "page", "expected_answer"),
    KNOWN_QUESTIONS,
)
def test_answers_known_questions_from_retrieved_context(
    monkeypatch,
    question,
    document_text,
    page,
    expected_answer,
):
    chunk = {
        "text": document_text,
        "metadata": {"source": "hospital-info.pdf", "page": page},
        "score": 0.9,
    }
    calls = {}

    def fake_retrieve(search_text, top_k):
        calls["retrieval"] = (search_text, top_k)
        return [chunk]

    def fake_generate(search_text, context, model):
        calls["generation"] = (search_text, context, model)
        assert document_text in context
        return expected_answer

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "_generate_answer", fake_generate)

    result = pipeline.answer_question(
        question,
        top_k=8,
        top_n=2,
        relevance_threshold=0.2,
        model="test-model",
    )

    assert calls["retrieval"] == (question, 8)
    assert calls["generation"][2] == "test-model"
    assert result["answer"] == expected_answer
    assert result["sources"] == [
        {"source": "hospital-info.pdf", "page": page, "score": 0.9}
    ]


def test_unknown_question_returns_safe_answer_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda question, top_k: [
            {
                "text": "General inpatient visiting hours are 4:00 PM to 7:00 PM.",
                "metadata": {"source": "hospital-info.pdf", "page": 1},
                "score": 0.98,
            }
        ],
    )

    def should_not_generate(*args, **kwargs):
        pytest.fail("The LLM should not answer from unrelated context.")

    monkeypatch.setattr(pipeline, "_generate_answer", should_not_generate)

    result = pipeline.answer_question("Does the hospital have an underwater café?")

    assert "not available" in result["answer"].casefold()
    assert "does not specify" in result["answer"].casefold()
    assert "contact the hospital" in result["answer"].casefold()
    assert result["sources"] == []


def test_icu_visitor_followup_retrieves_and_carries_explicit_limit(monkeypatch):
    document = (
        "ICU visiting hours are 5 PM to 6 PM daily. A maximum of one visitor "
        "may be present at a time."
    )
    captured = {}
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda question, top_k: [{
            "text": document,
            "metadata": {"source": "hospital-info.pdf", "page": 2},
            "score": 0.91,
        }],
    )

    def generate(question, context, model):
        captured["question"] = question
        captured["context"] = context
        return "No. The policy allows one visitor at a time, so two cannot be present simultaneously."

    monkeypatch.setattr(pipeline, "_generate_answer", generate)
    question = _knowledge_module.contextualize_question(
        "Can two people enter?",
        [{"role": "user", "content": "What are the ICU visiting hours?"}],
    )
    result = pipeline.answer_question(question)

    assert "ICU" in captured["question"]
    assert "one visitor" in captured["context"]
    assert "two cannot be present simultaneously" in result["answer"]
    assert result["sources"][0]["page"] == 2


def test_lab_hours_query_is_normalized_to_document_terms(monkeypatch):
    document = (
        "Diagnostic Laboratory. The laboratory provides routine and urgent "
        "diagnostic testing. Typical availability: Routine blood testing "
        "Mon-Sat, 7:00 AM-8:00 PM."
    )
    captured = {}

    def retrieve(search_text, top_k):
        captured["query"] = search_text
        return [{
            "text": document,
            "metadata": {"source": "hospital-info.pdf", "page": 2},
            "score": 0.9,
        }]

    def generate(question, context, model):
        captured["question"] = question
        assert document in context
        return "Routine blood testing is typically available Monday to Saturday, 7:00 AM to 8:00 PM."

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(pipeline, "_generate_answer", generate)

    result = pipeline.answer_question("What time lab open")

    assert "diagnostic laboratory" in captured["query"].casefold()
    assert "typical availability" in captured["query"].casefold()
    assert captured["question"] == "What time lab open"
    assert "7:00 AM" in result["answer"]
    assert result["sources"][0]["page"] == 2


def test_real_icu_followup_retrieval_keeps_explicit_limit_in_top_results(monkeypatch):
    total_chunks = collection_size()
    if not total_chunks:
        pytest.skip("Run scripts/ingest.py before the RAG retrieval checks.")

    query = _knowledge_module.contextualize_question(
        "how many people can go in?",
        [{"role": "user", "content": "What are the ICU visiting hours?"}],
    )
    def generate(question, context, model):
        assert "Maximum 1 visitor at a time" in context
        return "The ICU allows one visitor at a time."

    monkeypatch.setattr(pipeline, "_generate_answer", generate)
    result = pipeline.answer_question(query, top_k=total_chunks)

    assert "one visitor at a time" in result["answer"]
    assert result["sources"][0]["source"] == "hospital-info.pdf"
    assert result["sources"][0]["page"] == 1


def test_real_icu_two_person_followup_retrieves_limit_and_contact(monkeypatch):
    total_chunks = collection_size()
    if not total_chunks:
        pytest.skip("Run scripts/ingest.py before the RAG retrieval checks.")

    history = [
        {"role": "user", "content": "What are the ICU visiting hours?"},
        {"role": "assistant", "content": "ICU visiting hours are 5 PM to 6 PM daily."},
        {"role": "user", "content": "how many people can go in?"},
        {"role": "assistant", "content": "The ICU allows one visitor at a time."},
    ]
    query = _knowledge_module.contextualize_question(
        "is there any possibilites to take two persons", history
    )
    assert "main hospital number" in query

    def generate(question, context, model):
        context_lower = context.casefold()
        assert "maximum 1 visitor at a time" in context_lower
        assert "+92 300 555 0100" in context
        return (
            "The ICU limit is one visitor at a time, so two people cannot enter "
            "together under the stated policy. Please call +92 300 555 0100 "
            "to ask the hospital about your request."
        )

    monkeypatch.setattr(pipeline, "_generate_answer", generate)
    result = pipeline.answer_question(query, top_k=total_chunks)

    assert "cannot enter together" in result["answer"]
    assert "+92 300 555 0100" in result["answer"]
    assert result["sources"]


def test_real_general_ward_clarification_answer_retrieves_visiting_hours(monkeypatch):
    total_chunks = collection_size()
    if not total_chunks:
        pytest.skip("Run scripts/ingest.py before the RAG retrieval checks.")

    query = _knowledge_module.contextualize_question(
        "general ward",
        [
            {"role": "assistant", "content": "Which department or hospital service do you mean? Please name it so I can check its hours."},
        ],
    )
    assert query == "What are the general ward visiting hours?"

    def generate(question, context, model):
        assert "4:00 PM to 7:00 PM daily" in context
        return "General ward visiting hours are 4:00 PM to 7:00 PM daily."

    monkeypatch.setattr(pipeline, "_generate_answer", generate)
    result = pipeline.answer_question(query, top_k=total_chunks)

    assert "4:00 PM to 7:00 PM" in result["answer"]
    assert result["sources"]


def test_unknown_question_with_high_vector_score_is_rejected_without_keyword_evidence(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda question, top_k: [{
            "text": "General inpatient visiting hours are 4:00 PM to 7:00 PM daily.",
            "metadata": {"source": "hospital-info.pdf", "page": 1},
            "score": 0.99,
        }],
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_answer",
        lambda *args, **kwargs: pytest.fail("Unrelated context must not reach generation."),
    )
    result = pipeline.answer_question("Does the hospital have an underwater cafe?")
    assert "not available" in result["answer"].casefold()
    assert "does not specify" in result["answer"].casefold()
    assert "contact the hospital" in result["answer"].casefold()
    assert result["sources"] == []


def test_empty_retrieval_returns_no_sources_or_generated_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: [])
    monkeypatch.setattr(
        pipeline,
        "_generate_answer",
        lambda *args, **kwargs: pytest.fail("There is no context to answer from."),
    )

    result = pipeline.answer_question("What are the visiting hours?")

    assert "not available" in result["answer"].casefold()
    assert result["sources"] == []


@pytest.mark.parametrize(
    ("question", "expected_text"),
    [
        ("What time lab open", "routine blood testing"),
        ("What are the ICU visiting hours?", "maximum 1 visitor"),
        ("What are the visiting hours?", "4:00 pm"),
        ("Is the emergency department open 24 hours?", "24 hours a day"),
        ("What should I bring for an appointment?", "valid identification document"),
        ("What are the pharmacy hours?", "pharmacy support is available 24 hours"),
        ("What is the hospital privacy policy?", "confidential"),
    ],
)
def test_ingested_document_retrieves_relevant_text_and_page(
    monkeypatch,
    question,
    expected_text,
):
    total_chunks = collection_size()
    if not total_chunks:
        pytest.skip("Run scripts/ingest.py before the RAG retrieval checks.")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    retrieval_question = pipeline._normalize_retrieval_query(question)
    chunks = retrieve(retrieval_question, top_k=total_chunks)
    ranked = rerank_chunks(retrieval_question, chunks, top_n=3)

    match = next(
        (chunk for chunk in ranked if expected_text in chunk["text"].casefold()),
        None,
    )
    assert match is not None
    assert match["metadata"]["source"] == "hospital-info.pdf"
    assert isinstance(match["metadata"]["page"], int)
    assert isinstance(match["score"], float)


def test_unknown_question_with_real_index_stays_unanswered(monkeypatch):
    total_chunks = collection_size()
    if not total_chunks:
        pytest.skip("Run scripts/ingest.py before the RAG retrieval checks.")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        pipeline,
        "_generate_answer",
        lambda *args, **kwargs: pytest.fail("Unrelated chunks must not reach the LLM."),
    )

    result = pipeline.answer_question(
        "Does the hospital have an underwater cafe?",
        top_k=total_chunks,
    )

    assert "not available" in result["answer"].casefold()
    assert result["sources"] == []
