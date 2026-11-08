import pytest

from cognexa.knowledge.retrieval.result_aggregator import ResultAggregator


@pytest.fixture
def aggregator():
    return ResultAggregator(rrf_k=60)


def create_document(document_id, content, distance=0.5):
    return {
        "id": document_id,
        "content": content,
        "metadata": {},
        "distance": distance,
    }


def test_aggregate_multiple_queries(aggregator):
    query_1 = [
        create_document("A", "Document A"),
        create_document("B", "Document B"),
    ]

    query_2 = [
        create_document("B", "Document B"),
        create_document("C", "Document C"),
    ]

    results = aggregator.aggregate([query_1, query_2])

    assert len(results) == 3

    documents = {
        document["id"]: document
        for document in results
    }

    assert set(documents.keys()) == {"A", "B", "C"}


def test_aggregate_removes_duplicate_ids(aggregator):
    query_1 = [
        create_document("A", "Document A"),
    ]

    query_2 = [
        create_document("A", "Document A"),
    ]

    results = aggregator.aggregate([query_1, query_2])

    assert len(results) == 1
    assert results[0]["id"] == "A"


def test_rrf_score_is_calculated_correctly(aggregator):
    query_1 = [
        create_document("A", "Document A"),
        create_document("B", "Document B"),
    ]

    query_2 = [
        create_document("B", "Document B"),
        create_document("A", "Document A"),
    ]

    results = aggregator.aggregate([query_1, query_2])

    documents = {
        document["id"]: document
        for document in results
    }

    expected_score = (1 / 61) + (1 / 62)

    assert documents["A"]["rrf_score"] == pytest.approx(
        expected_score
    )

    assert documents["B"]["rrf_score"] == pytest.approx(
        expected_score
    )


def test_document_retrieved_at_high_rank_gets_higher_score(aggregator):
    query_1 = [
        create_document("A", "Document A"),
        create_document("B", "Document B"),
        create_document("C", "Document C"),
    ]

    results = aggregator.aggregate([query_1])

    assert results[0]["id"] == "A"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
    assert results[1]["rrf_score"] > results[2]["rrf_score"]


def test_document_tracks_retrieval_ranks(aggregator):
    query_1 = [
        create_document("A", "Document A"),
        create_document("B", "Document B"),
    ]

    query_2 = [
        create_document("B", "Document B"),
        create_document("A", "Document A"),
    ]

    results = aggregator.aggregate([query_1, query_2])

    document_a = next(
        document
        for document in results
        if document["id"] == "A"
    )

    assert document_a["retrieval_ranks"] == [
        {"query_index": 0, "rank": 1},
        {"query_index": 1, "rank": 2},
    ]

    assert document_a["matched_queries"] == [0, 1]


def test_document_keeps_minimum_distance(aggregator):
    query_1 = [
        create_document("A", "Document A", distance=0.5),
    ]

    query_2 = [
        create_document("A", "Document A", distance=0.2),
    ]

    results = aggregator.aggregate([query_1, query_2])

    assert results[0]["distance"] == pytest.approx(0.2)


def test_aggregate_empty_results(aggregator):
    results = aggregator.aggregate([])

    assert results == []


def test_aggregate_rejects_document_without_id(aggregator):
    invalid_document = {
        "content": "Document without ID",
    }

    with pytest.raises(ValueError):
        aggregator.aggregate([[invalid_document]])


def test_aggregate_rejects_non_dictionary_document(aggregator):
    with pytest.raises(TypeError):
        aggregator.aggregate([["invalid document"]])


def test_aggregator_rejects_invalid_rrf_k():
    with pytest.raises(ValueError):
        ResultAggregator(rrf_k=0)

    with pytest.raises(ValueError):
        ResultAggregator(rrf_k=-1)
