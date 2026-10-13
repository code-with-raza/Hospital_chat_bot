from app.rag.reranker import rerank_chunks


def test_reranker_moves_a_keyword_match_ahead_of_a_higher_vector_score():
    irrelevant = {
        "text": "The parking garage is open every day.",
        "metadata": {"source": "hospital-info.pdf", "page": 2},
        "score": 0.95,
    }
    relevant = {
        "text": "General ward visiting hours are 4:00 PM to 7:00 PM daily.",
        "metadata": {"source": "hospital-info.pdf", "page": 1},
        "score": 0.75,
    }

    ranked = rerank_chunks("What are the visiting hours?", [irrelevant, relevant])

    assert ranked[0]["text"] == relevant["text"]
    assert ranked[0]["score"] == relevant["score"]
    assert ranked[0]["metadata"] == relevant["metadata"]
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_reranker_respects_top_n_and_keeps_original_chunks_unchanged():
    chunks = [
        {"text": f"Pharmacy information {index}", "metadata": {"page": index}, "score": 0.8}
        for index in range(3)
    ]
    original = [dict(chunk) for chunk in chunks]

    ranked = rerank_chunks("pharmacy information", chunks, top_n=2)

    assert len(ranked) == 2
    assert all("rerank_score" in chunk for chunk in ranked)
    assert chunks == original


def test_reranker_returns_no_results_when_top_n_is_zero():
    assert rerank_chunks("visiting hours", [{"text": "visiting"}], top_n=0) == []
