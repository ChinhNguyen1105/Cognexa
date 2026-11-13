import pytest

from cognexa.assistant.answer_extractor import AnswerExtractor


class FakeReranker:
    """Mock CrossEncoder reranker for unit tests."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.last_query = None
        self.last_candidates = []

    def rerank(self, query, candidates):
        self.last_query = query
        self.last_candidates = candidates

        results = [
            {
                "index": index,
                "sentence": sentence,
                "score": self.scores.get(sentence, 0.0),
            }
            for index, sentence in enumerate(candidates)
        ]

        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )


@pytest.fixture
def context():
    return (
        "Python is a high-level programming language. "
        "Python supports object-oriented programming. "
        "Python is widely used in artificial intelligence."
    )


def test_extract_candidates_ranks_sentences_by_score(context):
    high_level = "Python is a high-level programming language."
    oop = "Python supports object-oriented programming."
    ai = "Python is widely used in artificial intelligence."

    reranker = FakeReranker(
        scores={
            high_level: 0.8,
            oop: 0.95,
            ai: 0.1,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=5,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="what paradigm does python support",
        context=context,
    )

    assert [item["sentence"] for item in result] == [
        oop,
        high_level,
        ai,
    ]

    assert [item["score"] for item in result] == [
        0.95,
        0.8,
        0.1,
    ]


def test_extract_candidates_passes_question_to_reranker(context):
    question = "what paradigm does python support"
    reranker = FakeReranker()

    extractor = AnswerExtractor(reranker=reranker)

    extractor.extract_candidates(
        question=question,
        context=context,
    )

    assert reranker.last_query == question


def test_extract_candidates_returns_empty_for_empty_context():
    reranker = FakeReranker()
    extractor = AnswerExtractor(reranker=reranker)

    result = extractor.extract_candidates(
        question="what is python",
        context="",
    )

    assert result == []
    assert reranker.last_query is None


def test_extract_candidates_rejects_empty_question():
    extractor = AnswerExtractor(reranker=FakeReranker())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        extractor.extract_candidates(
            question="",
            context="Python is a programming language.",
        )


def test_extract_candidates_rejects_whitespace_question():
    extractor = AnswerExtractor(reranker=FakeReranker())

    with pytest.raises(ValueError, match="Question cannot be empty"):
        extractor.extract_candidates(
            question="   ",
            context="Python is a programming language.",
        )


def test_extract_candidates_respects_max_candidates(context):
    reranker = FakeReranker(
        scores={
            "Python is a high-level programming language.": 0.8,
            "Python supports object-oriented programming.": 0.95,
            "Python is widely used in artificial intelligence.": 0.7,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=2,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="what is python",
        context=context,
    )

    assert len(result) == 2


def test_extract_candidates_respects_score_gap_threshold(context):
    high_level = "Python is a high-level programming language."
    oop = "Python supports object-oriented programming."
    ai = "Python is widely used in artificial intelligence."

    reranker = FakeReranker(
        scores={
            high_level: 0.8,
            oop: 0.95,
            ai: -3.5,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="what paradigm does python support",
        context=context,
    )

    assert [item["sentence"] for item in result] == [
        oop,
        high_level,
    ]


def test_extract_candidates_keeps_minimum_candidates(context):
    reranker = FakeReranker(
        scores={
            "Python is a high-level programming language.": 0.8,
            "Python supports object-oriented programming.": 0.95,
            "Python is widely used in artificial intelligence.": -10.0,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=1.0,
    )

    result = extractor.extract_candidates(
        question="what is python",
        context=context,
    )

    assert len(result) >= 2


def test_split_sentences_removes_document_headers():
    text = (
        "[Document 1 | Source: python.txt | Chunk: 0]\n"
        "Python is a high-level programming language.\n"
        "Python supports object-oriented programming."
    )

    result = AnswerExtractor._split_sentences(text)

    assert result == [
        "Python is a high-level programming language.",
        "Python supports object-oriented programming.",
    ]


def test_split_sentences_removes_duplicates():
    text = (
        "Python is a programming language. "
        "Python is a programming language."
    )

    result = AnswerExtractor._split_sentences(text)

    assert result == [
        "Python is a programming language.",
    ]


def test_split_sentences_returns_empty_for_empty_text():
    result = AnswerExtractor._split_sentences("")

    assert result == []


def test_init_rejects_invalid_min_candidates():
    with pytest.raises(ValueError, match="min_candidates"):
        AnswerExtractor(
            reranker=FakeReranker(),
            min_candidates=0,
        )


def test_init_rejects_max_candidates_smaller_than_minimum():
    with pytest.raises(ValueError, match="max_candidates"):
        AnswerExtractor(
            reranker=FakeReranker(),
            min_candidates=3,
            max_candidates=2,
        )


def test_init_rejects_negative_score_gap_threshold():
    with pytest.raises(ValueError, match="score_gap_threshold"):
        AnswerExtractor(
            reranker=FakeReranker(),
            score_gap_threshold=-1.0,
        )


def test_extract_candidates_ranks_by_question_and_aspect(context):
    class MultiQueryReranker:
        def rerank(self, query, candidates):
            scores = {
                "main question": [0.9, 0.2, 0.1],
                "aspect second": [0.1, 0.95, 0.2],
            }[query]
            return [{"index": i, "sentence": sentence, "score": scores[i]} for i, sentence in enumerate(candidates)]

    extractor = AnswerExtractor(MultiQueryReranker(), min_candidates=1, max_candidates=5)
    result = extractor.extract_candidates("main question", context, ["aspect second"])
    second = "Python supports object-oriented programming."
    candidate = next(item for item in result if item["sentence"] == second)
    assert candidate["question_score"] == 0.2
    assert candidate["aspect_scores"]["aspect second"] == 0.95


def test_missing_aspect_score_is_not_fabricated(context):
    extractor = AnswerExtractor(FakeReranker(), min_candidates=1)
    result = extractor.extract_candidates("main", context, ["one aspect"])
    assert all("one aspect" in item["aspect_scores"] for item in result)



def test_query_overlap_ratio_is_asymmetric():
    question = "python programming language"
    sentence = "python is a popular programming language"

    overlap = AnswerExtractor._query_overlap_ratio(
        question,
        sentence,
    )

    assert overlap == pytest.approx(1.0)


def test_query_overlap_ratio_returns_zero_without_content_tokens():
    overlap = AnswerExtractor._query_overlap_ratio(
        "the is a",
        "and or but",
    )

    assert overlap == 0.0


def test_ranking_adjustment_penalizes_query_repetition():
    question = "python programming language"
    sentence = "python programming language"

    adjustment, overlap = AnswerExtractor._ranking_adjustment(
        question,
        sentence,
    )

    assert overlap == pytest.approx(1.0)
    assert adjustment < 0.0


def test_ranking_adjustment_rewards_informative_sentence():
    question = "why does python support object oriented programming"
    sentence = (
        "Python supports object-oriented programming because "
        "it provides classes and inheritance."
    )

    adjustment, _ = AnswerExtractor._ranking_adjustment(
        question,
        sentence,
    )

    assert adjustment > 0.0


def test_jaccard_similarity_returns_one_for_identical_content():
    similarity = AnswerExtractor._jaccard_similarity(
        "python programming language",
        "python programming language",
    )

    assert similarity == pytest.approx(1.0)


def test_jaccard_similarity_uses_union_of_tokens():
    similarity = AnswerExtractor._jaccard_similarity(
        "python programming language",
        "python programming language tutorial",
    )

    assert similarity == pytest.approx(0.75)


def test_jaccard_similarity_returns_zero_for_empty_tokens():
    similarity = AnswerExtractor._jaccard_similarity(
        "the is a",
        "and or but",
    )

    assert similarity == 0.0


def test_jaccard_penalty_does_not_penalize_informative_sentence():
    question = "python programming language"
    sentence = (
        "Python programming language supports web development "
        "and machine learning applications."
    )

    adjustment, _ = AnswerExtractor._ranking_adjustment(
        question,
        sentence,
    )

    assert adjustment == 0.0


def test_jaccard_penalty_reduces_repetitive_sentence_score():
    question = "python programming language"
    sentence = "Python programming language"

    adjustment, _ = AnswerExtractor._ranking_adjustment(
        question,
        sentence,
    )

    assert adjustment == pytest.approx(-0.35)


def test_neighbor_expansion_includes_relevant_neighbor_below_min_score():
    seed = (
        "L2 regularization penalizes large model weights "
        "to control overfitting."
    )
    neighbor = (
        "L2 regularization is a regularization method "
        "that helps control overfitting."
    )
    unrelated = "The weather is sunny and warm today."

    context = f"{neighbor} {seed} {unrelated}"

    reranker = FakeReranker({
        seed: 0.9,
        neighbor: 0.1,
        unrelated: -0.5,
    })

    extractor = AnswerExtractor(
        reranker=reranker,
        min_score=0.5,
        neighbor_window=1,
        neighbor_similarity_threshold=0.15,
    )

    result = extractor.extract_candidates(
        question="How does L2 regularization control overfitting?",
        context=context,
    )

    result_sentences = [item["sentence"] for item in result]

    assert seed in result_sentences
    assert neighbor in result_sentences
    assert all(
        item.get("is_neighbor", False)
        for item in result
        if item["sentence"] == neighbor
    )


def test_neighbor_expansion_rejects_unrelated_sentence():
    seed = "L2 regularization penalizes large model weights."
    unrelated = "The weather is sunny and warm today."

    context = f"{seed} {unrelated}"

    reranker = FakeReranker({
        seed: 0.9,
        unrelated: 0.1,
    })

    extractor = AnswerExtractor(
        reranker=reranker,
        min_score=0.5,
        neighbor_window=1,
        neighbor_similarity_threshold=0.15,
    )

    result = extractor.extract_candidates(
        question="What does L2 regularization do?",
        context=context,
    )

    result_sentences = [item["sentence"] for item in result]

    assert seed in result_sentences
    assert unrelated not in result_sentences


def test_neighbor_expansion_does_not_add_duplicate_candidates():
    first = (
        "Regularization helps control overfitting "
        "during model training."
    )
    middle = (
        "Regularization methods help control overfitting "
        "in machine learning models."
    )
    last = (
        "Regularization techniques reduce overfitting "
        "during machine learning training."
    )

    context = f"{first} {middle} {last}"

    reranker = FakeReranker({
        first: 0.9,
        middle: 0.1,
        last: 0.8,
    })

    extractor = AnswerExtractor(
        reranker=reranker,
        min_score=0.5,
        neighbor_window=1,
        neighbor_similarity_threshold=0.15,
    )

    result = extractor.extract_candidates(
        question="How does regularization reduce overfitting?",
        context=context,
    )

    result_sentences = [item["sentence"] for item in result]

    assert result_sentences.count(middle) == 1
    assert len(result_sentences) == len(set(result_sentences))


def test_diversity_ranking_penalizes_redundant_information():
    first = (
        "Overfitting occurs when a model learns training data "
        "too closely, including noise."
    )
    repetitive = (
        "Overfitting occurs when a model learns training data "
        "too closely, including noise and accidental patterns."
    )
    informative = (
        "An overfitted model performs well on training data "
        "but poorly on unseen data."
    )

    candidates = [
        {
            "sentence": first,
            "score": 5.0,
            "ranking_score": 5.0,
        },
        {
            "sentence": repetitive,
            "score": 4.9,
            "ranking_score": 4.9,
        },
        {
            "sentence": informative,
            "score": 4.8,
            "ranking_score": 4.8,
        },
    ]

    result = AnswerExtractor._apply_diversity_ranking(candidates)

    assert result[0]["sentence"] == first
    assert result[1]["sentence"] == informative


def test_diversity_ranking_rewards_novel_information():
    first = "Overfitting occurs when a model learns training data too closely."
    second = (
        "An overfitted model performs well on training data "
        "but poorly on unseen data."
    )

    candidates = [
        {
            "sentence": first,
            "score": 5.0,
            "ranking_score": 5.0,
        },
        {
            "sentence": second,
            "score": 4.9,
            "ranking_score": 4.9,
        },
    ]

    result = AnswerExtractor._apply_diversity_ranking(candidates)

    assert result[0]["sentence"] == first
    assert result[1]["sentence"] == second
    assert result[1]["diversity_score"] > result[1]["ranking_score"]


def test_extract_candidates_keeps_multi_point_prevention_evidence_under_large_gap():
    sentences = [
        "Regularization techniques help reduce overfitting by discouraging overly complex models.",
        "Common approaches include L1 regularization, L2 regularization, dropout, and early stopping.",
        "Overfitting reduces the generalization ability of a model.",
    ]
    reranker = FakeReranker(
        scores={
            sentences[0]: 5.7,
            sentences[1]: -4.9,
            sentences[2]: 1.9,
        }
    )

    extractor = AnswerExtractor(
        reranker=reranker,
        min_candidates=2,
        max_candidates=10,
        score_gap_threshold=3.0,
    )

    result = extractor.extract_candidates(
        question="How can overfitting be reduced?",
        context=" ".join(sentences),
        aspect_queries=[{"aspect": "prevention", "question": "How can overfitting be reduced?"}],
    )

    texts = [item["sentence"] for item in result]
    assert sentences[0] in texts
    assert sentences[1] in texts
